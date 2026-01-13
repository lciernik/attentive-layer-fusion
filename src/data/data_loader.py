import argparse
from typing import Callable

import numpy as np
import torchvision.transforms as transforms
import webdataset as wds
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.data import build_dataset, get_dataset_collate_fn
from src.data.data_utils import GaussianNoise, get_fewshot_indices
from src.data.feature_combiner import BaseFeatureCombiner
from src.data.feature_datasets import (
    CombinedFeaturesDataset,
    FeatureDataset,
    Rep2RepFeaturesDataset,
)
from src.utils.tasks import Task
from src.utils.utils import load_features_targets


def adjust_batch_size(batch_size: int, num_samples: int, min_batches: int = 5) -> int:
    """Adjust the batch size to ensure that the number of batches is at least `min_batches`,
    while maximizing the size of the final batch.

    This function attempts to find the largest possible batch size such that:
      - The number of full batches is at least `min_batches`.
      - The final batch (if any) is as large as possible (ideally close to a full batch).
      - If the original batch size already yields at least `min_batches`, it is returned unchanged.

    Args:
        batch_size (int): The initial batch size.
        num_samples (int): The total number of samples in the dataset.
        min_batches (int, optional): The minimum number of batches required. Defaults to 5.

    Returns:
        int: The adjusted batch size.
    """

    def remainder_ok(remainder: int, batch_size: int) -> bool:
        """Check if the remainder is ok, i.e. close to a full batch."""
        return remainder == 0 or remainder >= 0.5 * batch_size

    nr_batches = num_samples // batch_size
    remainder = num_samples % batch_size
    if nr_batches >= min_batches and remainder_ok(remainder, batch_size):
        logger.info(f"Init. batch size ({batch_size}) was oki!")
        return batch_size

    best_batch_size = 1

    if nr_batches >= min_batches:
        min_batches = nr_batches

    max_batch_size = (num_samples // min_batches) + 1

    for candidate_batch_size in range(max_batch_size, 16, -1):
        remainder = num_samples % candidate_batch_size
        nr_full_batches = num_samples // candidate_batch_size
        if nr_full_batches < min_batches:
            continue
        if remainder_ok(remainder, candidate_batch_size):
            best_batch_size = candidate_batch_size
            break

    final_batches = num_samples // best_batch_size
    final_remainder = num_samples % best_batch_size

    logger.warning(
        f"Adjusted batch size from {batch_size} to {best_batch_size} for {num_samples} samples "
        f"({final_batches} batches of {best_batch_size} + final batch of {final_remainder})."
    )
    return best_batch_size


def count_samples_n_get_targets(dataset: Dataset) -> int:
    """Count the number of samples and get the targets of a dataset."""
    logger.info(f"Counting samples and getting targets.")
    targets = []
    for _, target in dataset:
        targets.append(target)
    return len(targets), np.array(targets)


def get_image_dataloader(
    args: argparse.Namespace,
    dataset_root: str,
    transform: Callable,
    pin_memory: bool = False,
) -> tuple[DataLoader | None, DataLoader]:
    """Get the image dataloader for the training, validation, and testing splits."""
    logger.info(f"> Load datasets and dataloaders for {args.dataset}. ")
    eval_dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform,
        split=args.split,  # by default this is the test split
        download=True,
        wds_cache_dir=args.wds_cache_dir,
    )

    if isinstance(eval_dataset, wds.WebDataset):
        nr_samples_eval, targets_eval = count_samples_n_get_targets(eval_dataset)
        eval_dataset = eval_dataset.with_length(nr_samples_eval)
        eval_dataset.targets = targets_eval
    else:
        nr_samples_eval = len(eval_dataset)

    collate_fn = get_dataset_collate_fn(args.dataset)

    logger.info(f"Dataset generation for split: {args.split}")
    try:
        logger.info(f"Dataset size: {len(eval_dataset)}")
    except TypeError:
        logger.info("IterableDataset has no len()")

    if hasattr(eval_dataset, "classes") and eval_dataset.classes:
        try:
            logger.info(f"Dataset classes: {eval_dataset.classes}")
            logger.info(f"Dataset number of classes: {len(eval_dataset.classes)}")
        except AttributeError:
            logger.info("Dataset has no classes.")

    # Get the dataloader for the split we want to evaluate on, by default this is the test split
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    # we also need the train and validation splits for linear probing.
    logger.info(f"Dataset generation for split: {args.train_split}")
    train_data_config = {
        "dataset_name": args.dataset,
        "root": dataset_root,
        "transform": transform,
        "split": args.train_split,
        "download": True,
    }
    train_dataset = build_dataset(**train_data_config)
    if train_dataset:
        if isinstance(train_dataset, wds.WebDataset):
            nr_samples_train, targets_train = count_samples_n_get_targets(train_dataset)
            train_dataset = train_dataset.with_length(nr_samples_train)
            train_dataset.targets = targets_train
            shuffle_train = False
        else:
            nr_samples_train = len(train_dataset)
            shuffle_train = True
        train_batch_size = adjust_batch_size(args.batch_size, nr_samples_train)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=shuffle_train and args.task != Task.FEATURE_EXTRACTION,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    else:
        train_dataloader = None
        nr_samples_train = None

    logger.info(f"> Train dataset size: {nr_samples_train}")
    logger.info(f"> Eval dataset size: {nr_samples_eval}")

    return train_dataloader, eval_dataloader, train_data_config


def get_feature_dl(
    feature_dir: str,
    batch_size: int,
    num_workers: int,
    fewshot_k: int,
    idxs: list[int] | None = None,
    load_train: bool = True,
    jitter_p: float = 0.5,
    normalize: bool = True,
    pin_memory: bool = False,
) -> tuple[DataLoader | None, DataLoader]:
    """Load the features from the feature_dir and return the dataloaders for training,
    validation, and testing.

    Args:
        feature_dir: The directory containing the features.
        batch_size: The batch size.
        num_workers: The number of workers.
        fewshot_k: The number of samples to use for few-shot learning.
        idxs: The indices of the samples to use for few-shot learning.
        load_train: Whether to load the training data.
        jitter_p: The probability of applying the Gaussian noise transform to a batch of features.
        normalize: Whether to normalize the features.
        pin_memory: Whether to pin the memory of the features.
    """
    features_test, targets_test = load_features_targets(feature_dir, split="test", normalize=normalize)
    feature_test_dset = FeatureDataset(features_test, targets_test, transform=None)

    feature_test_loader = DataLoader(
        feature_test_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if load_train:
        features, targets = load_features_targets(feature_dir, split="train", normalize=normalize)
        if fewshot_k < 0:  # if fewshot_k is -1, use the whole dataset
            train_features = features
            train_labels = targets
        else:
            if idxs is None:
                idxs = get_fewshot_indices(targets, fewshot_k)

            train_features = features[idxs]
            train_labels = targets[idxs]

        train_transform = transforms.Compose([GaussianNoise(mean=0.0, std=0.05, p=jitter_p)])

        feature_train_dset = FeatureDataset(train_features, train_labels, transform=train_transform)
        train_batch_size = adjust_batch_size(batch_size, len(train_features))

        feature_train_loader = DataLoader(
            feature_train_dset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    else:
        feature_train_loader = None

    return feature_train_loader, feature_test_loader


def get_combined_feature_dl(
    feature_dirs: list[str],
    batch_size: int,
    num_workers: int,
    fewshot_k: int,
    feature_combiner_cls: BaseFeatureCombiner,
    jitter_p: float = 0.5,
    normalize: bool = True,
    load_train: bool = True,
    pin_memory: bool = False,
) -> tuple[DataLoader | None, DataLoader]:
    """Load the features from the feature_dirs and return the dataloaders for training,
    validation, and testing.

    Args:
        feature_dirs: The directories containing the features.
        batch_size: The batch size.
        num_workers: The number of workers.
        fewshot_k: The number of samples to use for few-shot learning.
        feature_combiner_cls: The class of the feature combiner.
        jitter_p: The probability of applying the Gaussian noise transform to a batch of features.
        normalize: Whether to normalize the features.
        load_train: Whether to load the training data.
        pin_memory: Whether to pin the memory of the features.
    """
    if load_train:
        list_features, targets = load_features_targets(feature_dirs, split="train", normalize=normalize)

        if not all(len(feat) == len(list_features[0]) for feat in list_features):
            raise ValueError("Features of the different models have different number of samples.")

        if fewshot_k < 0:  # if fewshot_k is -1, use the whole dataset
            list_train_features = list_features
            train_labels = targets
        else:
            idxs = get_fewshot_indices(targets, fewshot_k)

            list_train_features = [features[idxs] for features in list_features]
            train_labels = targets[idxs]

        feature_combiner_train = feature_combiner_cls()
        train_transform = transforms.Compose([GaussianNoise(mean=0.0, std=0.05, p=jitter_p)])
        feature_train_dset = CombinedFeaturesDataset(
            list_train_features, train_labels, feature_combiner_train, transform=train_transform
        )
        train_batch_size = adjust_batch_size(batch_size, len(feature_train_dset))
        feature_train_loader = DataLoader(
            feature_train_dset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # TODO: Load a trained feature combiner, if neccessary # noqa: TD002, TD003
        feature_train_loader = None
        feature_combiner_train = None

    list_features_test, targets_test = load_features_targets(feature_dirs, split="test", normalize=normalize)

    feature_combiner_test = feature_combiner_cls(reference_combiner=feature_combiner_train)
    feature_test_dset = CombinedFeaturesDataset(
        list_features_test, targets_test, feature_combiner_test, transform=None
    )
    feature_test_loader = DataLoader(
        feature_test_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return feature_train_loader, feature_test_loader


def get_rep2rep_feature_dl(
    feature_dirs: list[str],
    batch_size: int,
    num_workers: int,
    fewshot_k: int = -1,
    load_train: bool = True,
    normalize: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader | None, DataLoader]:
    """Load the features from the feature_dir and return the dataloaders for training,
    validation, and testing.

    Args:
        feature_dirs: The directories containing the features.
        batch_size: The batch size.
        num_workers: The number of workers.
        fewshot_k: The number of samples to use for few-shot learning.
        load_train: Whether to load the training data.
        normalize: Whether to normalize the features.
    """
    if len(feature_dirs) != 2:
        raise ValueError("Only two feature directories are supported for rep2rep experiments.")

    if load_train:
        list_features, targets = load_features_targets(feature_dirs, split="train", normalize=normalize)

        if not all(len(feat) == len(list_features[0]) for feat in list_features):
            raise ValueError("Features of the different models have different number of samples.")

        if fewshot_k < 0:  # if fewshot_k is -1, use the whole dataset
            list_train_features = list_features
            train_labels = targets
        else:
            idxs = get_fewshot_indices(targets, fewshot_k)

            list_train_features = [features[idxs] for features in list_features]
            train_labels = targets[idxs]

        feature_train_dset = Rep2RepFeaturesDataset(list_train_features, train_labels)
        feature_train_loader = DataLoader(
            feature_train_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        feature_train_loader = None

    list_features_test, targets_test = load_features_targets(feature_dirs, split="test", normalize=normalize)
    feature_test_dset = Rep2RepFeaturesDataset(list_features_test, targets_test)
    feature_test_loader = DataLoader(
        feature_test_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return feature_train_loader, feature_test_loader
