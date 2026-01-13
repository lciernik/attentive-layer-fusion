import json
import os
from itertools import product

import numpy as np


def get_model_combinations(file_to_model_combinations):
    if os.path.isfile(file_to_model_combinations):
        with open(file_to_model_combinations, "r") as f:
            model_combinations = [[m.strip() for m in line.split(";")] for line in f if line.strip()]
    else:
        raise ValueError("The file does not exist", file_to_model_combinations)
    return model_combinations


def get_partition_based_on_train_samples(train_samples):
    if train_samples is None:
        return None
    elif train_samples <= 10000:
        partition = "gpu-5h"
    else:
        partition = "gpu-2d"
    return partition


def compute_memory_requirements(nr_models, ds_nr_samples, dim):
    nr_bytes = nr_models * ds_nr_samples * 2 * dim * 4
    mem = int(round(nr_bytes / (1024**3)) + 40)
    return mem


def load_models(file_path):
    with open(file_path, "r") as file:
        models = json.load(file)
    return models, len(models)


def count_nr_datasets(datasets_path):
    with open(datasets_path, "r") as f:
        return len(f.readlines())


def prepare_for_combined_usage(models):
    model_names = []
    sources = []
    model_parameters = []
    module_names = []
    for data in models.values():
        model_names.append(data["model_name"])
        sources.append(data["source"])
        model_parameters.append(data["model_parameters"])
        module_names.append(data["module_names"])
    return model_names, sources, model_parameters, module_names


def get_hyperparams(num_seeds=3, size="small"):
    if size == "small":
        hyper_params = dict(
            initial_lrs=["1e-3"],
            fewshot_ks=["-1"],
            epochs=["10"],
            reg_lambda="0.1",
            regularization=["weight_decay"],
            # attention_dropout=[(0.0, i) for i in [0.0, 0.1, 0.3]],
            attention_dropout=[(0.0, 0.1)],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "small_finetuning":
        hyper_params = dict(
            initial_lrs=["1e-3"],
            fewshot_ks=["-1"],
            epochs=["40"],
            reg_lambda="0.1",
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "eval_convergence":
        hyper_params = dict(
            initial_lrs=["1e-2", "1e-3", "1e-4"],
            fewshot_ks=["-1"],
            epochs=["50", "100", "200"],
            reg_lambda="0",
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "imagenet1k":
        hyper_params = dict(
            initial_lrs=["1e-1", "1e-2", "1e-3"],
            fewshot_ks=["-1"],
            epochs=["40"],
            reg_lambda="1e-2",
            regularization=["weight_decay"],
            attention_dropout=[(0.0, i) for i in [0.0, 0.1, 0.3]],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "IntermediateAAT":
        hyper_params = dict(
            initial_lrs=["1e-3"],
            fewshot_ks=["-1"],
            epochs=["40"],
            reg_lambda="1",
            regularization=["weight_decay"],
            attention_dropout=[(0.0, i) for i in [0.5]],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "imagenet1k-small":
        hyper_params = dict(
            initial_lrs=["1e-3"],
            fewshot_ks=["-1"],
            epochs=["40"],
            reg_lambda="1e-1",
            regularization=["weight_decay"],
            attention_dropout=[(0.0, i) for i in [0.1]],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "layerwise":
        hyper_params = dict(
            initial_lrs=["1e-2", "1e-3", "1e-4"],
            fewshot_ks=["-1"],
            epochs=["50"],
            reg_lambda="1e-2",
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "rep2rep":
        hyper_params = dict(
            # initial_lrs=["1e-2", "1e-3", "1e-4","1e-5"],
            initial_lrs=["1e-3", "1e-4"],
            fewshot_ks=["-1"],
            epochs=["50"],
            reg_lambda="1e-4",
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "rep2repFewshot":
        hyper_params = dict(
            # initial_lrs=["1e-2", "1e-3", "1e-4","1e-5"],
            initial_lrs=["1e-2", "1e-3", "1e-4"],
            fewshot_ks=["-1", "5", "10", "100"],
            epochs=["50"],
            reg_lambda="1e-4",
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "test":
        hyper_params = dict(
            initial_lrs=["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"],
            fewshot_ks=["-1"],
            epochs=["10"],
            reg_lambda=0.05,
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "n_heads":
        hyper_params = dict(
            initial_lrs=["1e-2", "1e-3", "1e-4"],
            fewshot_ks=["-1"],
            epochs=["10"],
            reg_lambda="1e-4",
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    else:
        hyper_params = dict(
            initial_lrs=["0.1", "0.01"],
            fewshot_ks=["-1", "5", "10", "100"],
            epochs=["10", "20", "30"],
            reg_lambda="0",
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    cols_for_array = ["fewshot_ks", "epochs", "regularization", "seeds"]
    num_jobs = np.prod([len(hyper_params[k]) for k in cols_for_array if k in hyper_params])
    return hyper_params, num_jobs


def format_path(path, num_samples_class, split):
    return path.format(num_samples_class=num_samples_class, split=split)


def parse_datasets(arg):
    if isinstance(arg, list):
        if len(arg) == 1 and os.path.isfile(arg[0]):
            with open(arg[0], "r") as f:
                datasets = [line.strip() for line in f if line.strip()]
        else:
            datasets = arg
    else:
        if os.path.isfile(arg):
            with open(arg, "r") as f:
                datasets = [line.strip() for line in f if line.strip()]
        else:
            datasets = [arg]
    return datasets
