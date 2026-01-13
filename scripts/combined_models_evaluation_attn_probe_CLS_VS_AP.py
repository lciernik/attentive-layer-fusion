import argparse
import json
import os
from pathlib import Path

from scripts.helper import (
    compute_memory_requirements,
    get_hyperparams,
    get_model_combinations,
    get_partition_based_on_train_samples,
    load_models,
    parse_datasets,
)
from scripts.project_location import CLUSTERING_ROOT, DATASETS_ROOT, FEATURES_ROOT, MODELS_ROOT, RESULTS_ROOT
from scripts.slurm import run_job

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models_config", type=str, default="./scripts/configs/models_config_single_model_layer_combination.json"
)
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default="./scripts/configs/webdataset_configs/webdatasets_experiments.txt",
    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing dataset names.",
)

parser.add_argument(
    "--models_combination",
    nargs="+",
    type=str,
    default=[
        "./scripts/configs/2_and_4_layers_CLD_AP/model_combinations_two_and_four_layers_equally_spaced_B.txt",
        "./scripts/configs/all_layers_CLS_AP/model_combinations_layers_all_blocks_cls_ap_B.txt",
    ],
    help="File containing model combinations to evaluate with an attentive probe.",
)

args = parser.parse_args()

MODELS_CONFIG = args.models_config


with Path("./scripts/configs/webdataset_configs/dataset_info.json").open("r") as f:
    ds_info = json.load(f)

if __name__ == "__main__":
    if not isinstance(args.models_combination, list):
        args.models_combination = [args.models_combination]

    prepared_datasets = sorted(set(parse_datasets(args.datasets)))

    ## Prepare variables shared by all model combinations
    models, n_models = load_models(MODELS_CONFIG)
    hyper_params, num_jobs = get_hyperparams(num_seeds=1, size="imagenet1k-small")
    val_proportion = 0.2
    dim_alignment = "zero_padding"

    for file_to_model_combinations in args.models_combination:
        model_combinations = get_model_combinations(file_to_model_combinations)

        for DATASET in prepared_datasets:
            ds_nr_samples = ds_info[DATASET]["nr_train_samples"]
            partition = get_partition_based_on_train_samples(ds_nr_samples)
            if partition is None:
                print(f"\n\nDataset {DATASET} has no train samples. Continuing...\n\n")
                continue

            for model_set in model_combinations:
                for proj_drop, att_drop in hyper_params["attention_dropout"]:
                    assert all([key.split("@")[0] in models.keys() for key in model_set])

                    dim = max([models[mid.split("@")[0]]["embedding_dim"] for mid in model_set])

                    AP_models = [m for m in model_set if "_ap@" in m]
                    CLS_models = [m for m in model_set if "_cls@" in m]

                    for curr_model_set in [AP_models, CLS_models]:
                        model_keys = " ".join(curr_model_set)
                        nr_models = len(curr_model_set)
                        n_heads = len(curr_model_set)
                        feature_combiner = "stacked_zero_pad"

                        mem = compute_memory_requirements(nr_models, ds_nr_samples, dim)

                        print(
                            f"\nRunning attentive probe for:\n{curr_model_set=},\n{proj_drop=} dropout,"
                            f"\n{att_drop=} dropout,\n{n_heads=} heads,\n{DATASET=} datasets,\nand {mem}GB memory\n"
                        )

                        job_cmd = f"""python src/cli.py --dataset {DATASET} \
                                        --dataset_root {Path(DATASETS_ROOT).absolute()} \
                                        --feature_root {Path(FEATURES_ROOT).absolute()} \
                                        --model_root {Path(MODELS_ROOT).absolute()} \
                                        --output_root {Path(RESULTS_ROOT).absolute()} \
                                        --clustering_root {Path(CLUSTERING_ROOT).absolute()} \
                                        --task=attentive_probe \
                                        --mode=combined_models \
                                        --feature_combiner {feature_combiner} \
                                        --model_key {model_keys} \
                                        --models_config_file {MODELS_CONFIG} \
                                        --batch_size=2048 \
                                        --dim {dim} \
                                        --num_heads {n_heads} \
                                        --dimension_alignment {dim_alignment} \
                                        --fewshot_k {" ".join(hyper_params["fewshot_ks"])} \
                                        --initial_lr {" ".join(hyper_params["initial_lrs"])} \
                                        --epochs {" ".join(hyper_params["epochs"])} \
                                        --reg_lambda {hyper_params["reg_lambda"]} \
                                        --regularization {" ".join(hyper_params["regularization"])} \
                                        --train_split train \
                                        --test_split test \
                                        --val_proportion {val_proportion} \
                                        --seed {" ".join(hyper_params["seeds"])} \
                                        --attention_dropout {proj_drop} {att_drop} \
                                        --grad_norm_clip 5 \
                                        --jitter_p 0.5 \
                                        --skip_existing
                        """

                        run_job(
                            job_name=f"attn_probe_with_n{nr_models}",
                            job_cmd=job_cmd,
                            partition=partition,
                            log_dir=f"{Path(RESULTS_ROOT).absolute()}/logs",
                            num_jobs_in_array=num_jobs,
                            mem=mem,
                        )
