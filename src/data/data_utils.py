import os

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageNet

from src.data.builder import build_dataset, get_dataset_collection, get_dataset_collection_from_file
from src.data.constants import probe_dataset_map
from src.data.feature_datasets import CombinedFeaturesDataset, FeatureDataset
from src.utils.utils import as_list


class GaussianNoise:
    """Gaussian noise transform."""

    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.5) -> None:
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian noise transform to the tensor."""
        if torch.rand(1).item() < self.p:
            return tensor + (torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean)
        return tensor


def get_fewshot_indices(targets: torch.Tensor, fewshot_k: int) -> list[int]:
    """Get the indices of the features that are use for training the linear probe"""
    length = len(targets)
    perm = [p.item() for p in torch.randperm(length)]
    idxs = []
    counts = {}
    num_classes = 0

    for p in perm:
        target = targets[p].item()
        if target not in counts:
            counts[target] = 0
            num_classes += 1

        if fewshot_k < 0 or counts[target] < fewshot_k:
            counts[target] += 1
            idxs.append(p)

    for c, val in counts.items():
        if fewshot_k > 0 and val != fewshot_k:
            raise ValueError(
                f"insufficient data for eval with {fewshot_k} samples per class, only {val} samples for class {c}"
            )

    return idxs


class SubsetWithTargets(Subset):
    """Simple dataset wrapper with targets."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        super().__init__(dataset, indices)
        self.targets = np.array([dataset.targets[i] for i in indices])

    @property
    def feature_dims(self) -> int:
        """Get the feature dimensions of the dataset."""
        return self.dataset.feature_dims


def create_train_val_loaders(
    train_loader: DataLoader,
    train_dataset_config: dict | None = None,
    val_proportion: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Creates train and validation loaders from a given train loader.

    Supports:
    - WebDatasets (with .select() method):
        - If WebDataset has a .targets attribute, uses stratified splitting
        - Otherwise, uses hash-based splitting
    - FeatureDataset/CombinedFeaturesDataset instances: uses stratified splitting

    Args:
        train_loader (DataLoader): The training data loader to split.
        train_dataset_config (dict | None): The configuration of the train dataset.
        val_proportion (float, optional): Proportion of the dataset to include in the validation set. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        pin_memory (bool, optional): Whether to pin memory in DataLoader. Defaults to True.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing:
            - train_loader: DataLoader for the training split
            - val_loader: DataLoader for the validation split
    """
    train_dataset = train_loader.dataset
    collate_fn = getattr(train_loader, "collate_fn", None)

    # Check if dataset is a WebDataset (has select method)
    if hasattr(train_dataset, "select"):
        # WebDataset path
        if hasattr(train_dataset, "targets"):
            # WebDataset with targets - use stratified splitting
            targets = np.array(train_dataset.targets)
            total_samples = len(targets)

            train_indices, val_indices = train_test_split(
                np.arange(total_samples),
                test_size=val_proportion,
                stratify=targets,
                random_state=seed,
            )
            train_indices_set = set(train_indices)
            val_indices_set = set(val_indices)

            logger.info(f"Separating train dataset into train and validation sets using stratified splitting.")

            # Create selector functions with modulo to handle multiple epochs
            train_counter = [0]

            def is_train(sample):
                """Select sample if its index is in train_indices."""
                idx = train_counter[0] % total_samples
                train_counter[0] += 1
                return idx in train_indices_set

            val_counter = [0]

            def is_val(sample):
                """Select sample if its index is in val_indices."""
                idx = val_counter[0] % total_samples
                val_counter[0] += 1
                return idx in val_indices_set

            # Build NEW dataset instances with selectors applied
            tmp_train_dataset = (
                build_dataset(**train_dataset_config).select(is_train).shuffle(min(len(train_indices), 10_000))
            )
            tmp_train_dataset = tmp_train_dataset.with_length(len(train_indices))
            tmp_train_dataset.targets = targets[train_indices]  # Subset of targets

            tmp_val_dataset = (
                build_dataset(**train_dataset_config).select(is_val).shuffle(min(len(val_indices), 10_000))
            )
            tmp_val_dataset = tmp_val_dataset.with_length(len(val_indices))
            tmp_val_dataset.targets = targets[val_indices]  # Subset of targets

            logger.info(f"Train dataset size: {len(tmp_train_dataset)}")
            logger.info(f"Validation dataset size: {len(tmp_val_dataset)}")
            shuffle_train = False
        else:
            # WebDataset without targets - use hash-based splitting
            def is_train(sample):
                """Deterministic split based on sample key."""
                # key = sample.get("__key__", str(hash(str(sample))))
                key = str(hash(str(sample)))
                return hash(key + str(seed)) % 100 >= int(val_proportion * 100)

            def is_val(sample):
                """Deterministic split based on sample key."""
                # key = sample.get("__key__", str(hash(str(sample))))
                key = str(hash(str(sample)))
                return hash(key + str(seed)) % 100 < int(val_proportion * 100)

            tmp_train_dataset = train_dataset.select(is_train)
            tmp_val_dataset = train_dataset.select(is_val)
            shuffle_train = False
    elif isinstance(train_dataset, (FeatureDataset, CombinedFeaturesDataset, ImageNet)):
        # FeatureDataset or CombinedFeaturesDataset path
        targets = np.array(train_dataset.targets)
        train_indices, val_indices = train_test_split(
            np.arange(targets.shape[0]),
            test_size=val_proportion,
            stratify=targets,
            random_state=seed,
        )
        tmp_train_dataset = SubsetWithTargets(train_dataset, indices=train_indices)
        tmp_val_dataset = SubsetWithTargets(train_dataset, indices=val_indices)
        shuffle_train = True
    else:
        raise ValueError(
            "Dataset must be either a WebDataset (with .select() method) "
            "or a FeatureDataset/CombinedFeaturesDataset instance."
        )

    def check_batch_size(dataset: Dataset) -> int:
        batch_size = train_loader.batch_size
        if len(dataset) % batch_size == 1:
            batch_size += 1
        return batch_size

    tmp_train_batch_size = check_batch_size(tmp_train_dataset)
    tmp_val_batch_size = check_batch_size(tmp_val_dataset)

    tmp_train_loader = DataLoader(
        tmp_train_dataset,
        batch_size=tmp_train_batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle_train,
        num_workers=train_loader.num_workers,
        collate_fn=collate_fn,
    )
    tmp_val_loader = DataLoader(
        tmp_val_dataset,
        batch_size=tmp_val_batch_size,
        pin_memory=pin_memory,
        shuffle=False,
        num_workers=train_loader.num_workers,
        collate_fn=collate_fn,
    )
    return tmp_train_loader, tmp_val_loader


def get_list_of_datasets(base):
    datasets = []
    dataset_collection = get_dataset_collection()
    for name in as_list(base.dataset):
        if os.path.isfile(name):
            # If path, read file, each line is a dataset name
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            # if part of `dataset_collection`, retrieve from it
            datasets.extend(dataset_collection[name])
        else:
            # if not, assume it is simply the name of the dataset
            datasets.append(name)
    return datasets


def map_to_probe_dataset(dataset: str) -> str:
    if dataset in probe_dataset_map:
        return probe_dataset_map[dataset]
    return dataset


def check_force_train(dataset: str, force_train: bool) -> bool:
    if dataset in probe_dataset_map and force_train:
        return False
    return force_train


def prepare_ds_name(dataset: str) -> str:
    dataset = dataset.replace("/", "_")
    return dataset
