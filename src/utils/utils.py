import argparse
import json
import os
import random
import sqlite3
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger

from src.utils.model_mapping import split_model_module
from src.utils.tasks import Task


def as_list(x):
    if not x:
        return []
    return [x] if not isinstance(x, list) else x


def get_subset_data(data: torch.Tensor, subset_indices: List[int]) -> torch.Tensor:
    return data[subset_indices]


def load_features(
    feature_root: str,
    model_id: Optional[str] = None,
    split: str = "train",
    subset_indices: Optional[List[int]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    model_dir = Path(feature_root)
    if model_id:
        subdir = "/".join(split_model_module(model_id, use_abbrev=False, always_return_tuple=False))
        model_dir = model_dir / subdir
    features = torch.load(os.path.join(model_dir, f"features_{split}.pt"))

    logger.info(f"Loaded features for {model_id} with shape {features.shape}")

    if subset_indices:
        features = get_subset_data(features, subset_indices)
        logger.info(f"Loaded subset of features for {model_id} with shape {features.shape}")
    if normalize:
        features = F.normalize(features, dim=1, p=2)
    return features


def load_targets(
    feature_root: str,
    model_id: Optional[str] = None,
    split: str = "train",
    subset_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    model_dir = Path(feature_root)
    if model_id:
        model_dir = model_dir / "/".join(split_model_module(model_id, use_abbrev=False, always_return_tuple=False))
    targets = torch.load(model_dir / f"targets_{split}.pt")
    if subset_indices:
        targets = get_subset_data(targets, subset_indices)
        logger.info(f"Loaded subset of features for {model_id} with shape {targets.shape}")

    return targets


def check_equal_targets(list_targets: List[torch.Tensor]) -> bool:
    if len(list_targets) > 1:
        first_targets = list_targets[0]
        for curr_target in list_targets[1:]:
            if not (first_targets == curr_target).all().item():
                return False
    return True


def load_features_targets(
    feature_root: str | list[str],
    model_id: Optional[str] = None,
    split: str = "train",
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(feature_root, list):
        features = [load_features(f, model_id, split, normalize=normalize) for f in feature_root]
        targets = [load_targets(f, model_id, split) for f in feature_root]
        if not check_equal_targets(targets):
            raise ValueError("Not all targets are equal.")
        targets = targets[0]
    else:
        features = load_features(feature_root, model_id, split, normalize=normalize)
        targets = load_targets(feature_root, model_id, split)
    print("Loaded all features and targets")

    return features, targets


## Check if features exist for all models
def check_models(feature_root, model_ids, split):
    prev_model_ids = model_ids
    model_ids = sorted(
        [
            mid
            for mid in model_ids
            if (
                Path(feature_root)
                / "/".join(split_model_module(mid, use_abbrev=False, always_return_tuple=False))
                / f"features_{split}.pt"
            ).exists()
        ]
    )

    if len(set(prev_model_ids)) != len(set(model_ids)):
        logger.info(f"Features do not exist for the following models: {set(prev_model_ids) - set(model_ids)}")
        logger.info("Removing the above models from the list of models for distance computation.")

    # Check if enough remaining models to compute distance matrix
    assert len(model_ids) > 1, "At least two models are required for distance computation"

    return model_ids


def single_option_to_multiple_datasets(cur_option: List[str], datasets: List[str], name: str) -> List[str]:
    cur_len = len(cur_option)
    ds_len = len(datasets)
    if cur_len != ds_len:
        # If user wants to use same value for all datasets
        if cur_len == 1:
            return [cur_option[0]] * ds_len
        else:
            raise ValueError(f"The incommensurable number of {name}")
    else:
        return cur_option


def get_train_val_splits(
    train_split: Union[str, List[str]],
    val_proportion: Union[float, List[float]],
    datasets: List[str],
) -> Dict[str, Dict[str, Optional[Union[str, float]]]]:
    train_splits = as_list(train_split)
    train_splits = single_option_to_multiple_datasets(train_splits, datasets, "train_split")
    proportions = None
    if val_proportion is not None:
        proportions = as_list(val_proportion)
        proportions = single_option_to_multiple_datasets(proportions, datasets, "val_proportion")

    dataset_info = {}
    for i in range(len(datasets)):
        dataset_info[datasets[i]] = {
            "train_split": train_splits[i],
            "proportion": proportions[i] if proportions is not None else None,
        }
    return dataset_info


def world_info_from_env():
    # from openclip
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def all_paths_exist(list_of_paths: list[str]) -> bool:
    """Check if all paths exist."""
    return all(Path(p).exists() for p in list_of_paths)


def set_all_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)


def prepare_device(distributed: bool) -> str:
    if torch.cuda.is_available():
        if distributed:
            local_rank, rank, world_size = world_info_from_env()
            device = "cuda:%d" % local_rank
            torch.cuda.set_device(device)
        else:
            device = "cuda"
        return device
    else:
        return "cpu"


def get_combination(
    fewshot_ks: List[int],
    epochs: List[int],
    seeds: List[int],
    regularization: List[str],
    get_all: bool = False,
) -> Tuple[Union[List[Tuple[int, int, int, str]], Tuple[int, int, int, str]], Optional[int]]:
    combs = []
    combs.extend(
        list(
            product(
                fewshot_ks,
                epochs,
                seeds,
                regularization,
            )
        )
    )
    if get_all:
        return combs, None
    else:
        comb_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        return combs[comb_idx], comb_idx


def get_list_of_models(
    base: argparse.Namespace,
) -> List[Tuple[str, str, dict, str, str, str]]:
    """Get list of models and config to evaluate."""
    models = as_list(base.model)
    srcs = as_list(base.model_source)
    params = as_list(base.model_parameters)
    module_names = as_list(base.module_names)
    feature_alignments = as_list(base.feature_alignment)
    model_keys = as_list(base.model_key)

    assert len(models) == len(srcs), "The number of model_source should be the same as the number of models"
    assert len(models) == len(params), "The number of model_parameters should be the same as the number of models"
    assert len(models) == len(module_names), "The number of module_names should be the same as the number of models"
    assert len(models) == len(feature_alignments), (
        "The number of feature_alignment should be the same as the number of models"
    )
    assert len(models) == len(model_keys), "The number of model_key should be the same as the number of models"

    models_w_config = list(zip(models, srcs, params, module_names, feature_alignments, model_keys))
    # TODO: Not clear if this is also the case if we have multiple representations of the same model
    if base.task.value != "rep2rep":  # For rep2rep the ordering is important!
        models_w_config = sorted(models_w_config, key=lambda x: x[-1])
    return models_w_config


def reduce_list_of_models(
    models_w_config: List[Tuple[str, str, dict, str, str, str]], new_model_set: List[str]
) -> List[Tuple[str, str, dict, str, str, str]]:
    # Reduce the list of models to only those that are in the new_model_set, Ensure that the module name is in model key
    reduced_models = []
    model_ids = [m[5] for m in models_w_config]
    for model_with_module in new_model_set:
        base_model, module = split_model_module(model_with_module, use_abbrev=False, always_return_tuple=True)
        # Check if the model is in the models_w_config
        idx = model_ids.index(base_model) if base_model in model_ids else -1
        if idx == -1:
            raise ValueError(f"Model {base_model} not found in the models_w_config.")
        added_model_config = list(models_w_config[idx])
        added_model_config[3] = module if module is not None else added_model_config[3]
        added_model_config[5] = model_with_module
        reduced_models.append(tuple(added_model_config))
    return reduced_models


def make_results_df(exp_args: argparse.Namespace, model_ids: List[str], metrics: Dict[str, float]) -> pd.DataFrame:
    results_current_run = {}

    # experiment config
    results_current_run["task"] = exp_args.task.value
    results_current_run["mode"] = exp_args.mode
    results_current_run["combiner"] = (
        exp_args.feature_combiner
        if (exp_args.task == Task.LINEAR_PROBE or exp_args.task == Task.ATTENTIVE_PROBE)
        and exp_args.mode == "combined_models"
        else None
    )
    # dataset
    results_current_run["dataset"] = exp_args.dataset
    results_current_run["feature_normalization"] = exp_args.normalize
    results_current_run["feature_alignment"] = json.dumps(exp_args.feature_alignment)
    results_current_run["train_split"] = exp_args.train_split
    results_current_run["val_proportion"] = exp_args.val_proportion
    results_current_run["test_split"] = exp_args.split
    if exp_args.task == Task.ATTENTIVE_PROBE:
        results_current_run["dim"] = exp_args.dim
        results_current_run["num_heads"] = exp_args.num_heads
        results_current_run["dimension_alignment"] = exp_args.dimension_alignment
        results_current_run["project_all_models"] = exp_args.always_project
        results_current_run["attention_dropout"] = json.dumps(exp_args.attention_dropout)
    results_current_run["jitter_p"] = exp_args.jitter_p
    # model(s)
    results_current_run["model_ids"] = json.dumps(model_ids)
    results_current_run["model"] = json.dumps(exp_args.model)
    results_current_run["model_source"] = json.dumps(exp_args.model_source)
    results_current_run["model_parameters"] = json.dumps(exp_args.model_parameters)
    results_current_run["module_names"] = json.dumps(exp_args.module_names)
    results_current_run["num_clusters"] = exp_args.num_clusters if exp_args.num_clusters != -1 else None
    results_current_run["clustering_similarity_method"] = (
        exp_args.clustering_similarity_method if exp_args.num_clusters != -1 else None
    )
    # hyperparameters
    results_current_run["fewshot_k"] = exp_args.fewshot_k
    results_current_run["epochs"] = exp_args.epochs
    results_current_run["batch_size"] = exp_args.batch_size
    results_current_run["seed"] = exp_args.seed
    results_current_run["regularization"] = exp_args.regularization
    results_current_run["use_class_weights"] = exp_args.use_class_weights
    results_current_run["grad_norm_clip"] = getattr(exp_args, "grad_norm_clip", None)
    # sae-training
    results_current_run["final_lr"] = getattr(exp_args, "final_lr", None)
    results_current_run["warmup_epochs"] = getattr(exp_args, "warmup_epochs", None)
    results_current_run["grad_clip"] = getattr(exp_args, "grad_clip", None)
    results_current_run["loss_type"] = getattr(exp_args, "loss_type", None)
    results_current_run["patience"] = getattr(exp_args, "patience", None)
    results_current_run["val_step"] = getattr(exp_args, "val_step", None)
    results_current_run["min_delta"] = getattr(exp_args, "min_delta", None)
    results_current_run["extract_train"] = getattr(exp_args, "extract_train", None)
    results_current_run["extract_test"] = getattr(exp_args, "extract_test", None)
    results_current_run["use_wandb"] = getattr(exp_args, "use_wandb", None)

    # rep2rep
    results_current_run["rep_loss"] = exp_args.rep_loss

    # Flatten metrics recursively
    def flatten_metrics(curr_metrics, parent_key=""):
        new_metrics = {}

        for k, v in curr_metrics.items():
            new_key = f"{parent_key.replace('_metrics', '')}_{k}" if parent_key else k  # Combine keys with '_'

            if isinstance(v, dict):
                new_metrics.update(flatten_metrics(v, new_key))  # Recursive call
            else:
                new_metrics[new_key] = v  # Base case

        return new_metrics

    flattened_metrics = flatten_metrics(metrics)
    for key, value in flattened_metrics.items():
        if key in results_current_run:
            continue
        results_current_run[key] = value

    return pd.DataFrame(results_current_run, index=range(1))


def save_results(
    args: argparse.Namespace,
    model_ids: List[str],
    metrics: Dict[str, float],
    out_path: str,
    fn: str = "results.json",
) -> None:
    """Save the results to json file."""
    results_current_run = make_results_df(exp_args=args, model_ids=model_ids, metrics=metrics)
    if len(results_current_run) == 0:
        raise ValueError("results_current_run had no entries")

    results_current_run.to_json(os.path.join(out_path, fn), default_handler=str)


def check_existing_results(out_path: str, fn: str = "results.json") -> bool:
    return os.path.exists(os.path.join(out_path, fn))


def get_base_evaluator_args(
    args: argparse.Namespace,
    feature_dirs: List[str],
    model_dirs: List[str],
    results_dir: str,
) -> Dict[str, Any]:
    probe_args = {
        "epochs": args.epochs,
        "reg_lambda": args.reg_lambda,
        "regularization": args.regularization,
        "device": args.device,
        "use_class_weights": args.use_class_weights,
        "grad_norm_clip": args.grad_norm_clip,
        "freeze_premodel": args.freeze_premodel,
    }
    base_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lrs": args.initial_lr,
        "probe_args": probe_args,
        "seed": args.seed,
        "fewshot_k": args.fewshot_k,
        "feature_dirs": feature_dirs,
        "model_dirs": model_dirs,
        "results_dir": results_dir,
        "normalize": args.normalize,
        "val_proportion": args.val_proportion,
        "force_train": args.force_train,
        "jitter_p": args.jitter_p,
        "reg_lambda_bounds": args.reg_lambda_bounds,
    }
    return base_kwargs


def retrieve_model_dataset_results(base_path_exp: str | Path, allow_db_results: bool = True) -> pd.DataFrame:
    path = Path(base_path_exp)
    dfs = []
    for fn in path.rglob("**/results.json"):
        df = pd.read_json(fn)
        dfs.append(df)

    if len(dfs) == 0:
        if allow_db_results:
            # backward compatibility
            bak_fn = path / "results.db"
            if bak_fn.is_file():
                logger.info(f"Did not find any results.json files. Trying to load data from {bak_fn}")
                try:
                    conn = sqlite3.connect(bak_fn)
                    df = pd.read_sql('SELECT * FROM "results"', conn)
                    conn.close()
                except pd.errors.DatabaseError as e:
                    logger.error(f"Tried to extract data from {path=}, but got Error: {e}")
                    raise e

                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = df[col].apply(json.loads)
            else:
                raise FileNotFoundError(
                    f"No results found for in {base_path_exp=} (neither .json files not results.db)"
                )
        else:
            raise FileNotFoundError(f"No results found for in {base_path_exp=} (no .json files)")
    else:
        df = pd.concat(dfs).reset_index(drop=True)

    logger.info(f"Found {len(df)} results in {base_path_exp=}")
    return df


def check_single_instance(param: list[Any] | Any, param_name: str) -> Any | list[Any]:
    if isinstance(param, list):
        if len(param) > 1:
            raise ValueError(f"Only supports a single {param_name} expected.")
        return param[0]
    return param


def check_feature_existence(
    feature_dir: str | Path, module_names: list[str] | None = None, check_train: bool = True, check_test: bool = True
) -> bool:
    """For a given feature directory, check if the features and targets exist for the given module names.
    If no module names are given, check if the features and targets exist in the feature directory.
    """
    if isinstance(feature_dir, str):
        feature_dir = Path(feature_dir)

    if not feature_dir.exists():
        feature_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Create path to store features: {feature_dir}")
        return False

    filenames_to_check = []
    if check_test:
        filenames_to_check += ["features_test.pt", "targets_test.pt"]
    if check_train:
        filenames_to_check += ["features_train.pt", "targets_train.pt"]

    if not filenames_to_check:
        raise ValueError("No filenames to check.")

    if module_names is not None:
        feature_dirs = [feature_dir / module_name for module_name in module_names]
    else:
        feature_dirs = [feature_dir]

    all_exist = True
    for curr_feature_dir in feature_dirs:
        for filename in filenames_to_check:
            curr_file = curr_feature_dir / filename
            if not curr_file.exists():
                all_exist = False
                logger.error(f"File {filename} is missing in {curr_feature_dir}.")
                break
    return all_exist


def load_model_from_file(
    model_path: Union[str, Path], device: str, use_data_parallel: bool = False
) -> torch.nn.Module:
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path=}")
    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    if use_data_parallel and torch.cuda.is_available():
        logger.info("> Using data parallel")
        model = torch.nn.DataParallel(model, device_ids=[x for x in range(torch.cuda.device_count())])
    elif not use_data_parallel and isinstance(model, torch.nn.DataParallel):
        model = model.module

    return model


def robust_to_device(x: Union[torch.Tensor, list, tuple], device: str) -> Union[torch.Tensor, list, tuple]:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [xi.to(device) for xi in x]
    elif isinstance(x, tuple):
        return tuple(xi.to(device) for xi in x)
    else:
        raise ValueError(f"Unsupported input type: {type(x)}")
