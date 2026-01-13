import argparse
from pathlib import Path


import pandas as pd


from scripts.helper import (
    compute_memory_requirements,
    get_partition_based_on_train_samples,
    load_models,
    parse_datasets,
)
from scripts.project_location import DATASETS_ROOT, FEATURES_ROOT, RESULTS_ROOT
from scripts.slurm import run_job

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models_config",
    type=str,
    # default="./scripts/configs/models_config_single_model_layer_combination.json"
    default="./scripts/configs/models_config_single_model_layer_combination_only_all_tokens.json",
)
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    # default="./scripts/configs/webdataset_configs/webdatasets_part_one_experiments.txt",
    default="./scripts/configs/webdataset_configs/webdatasets_all_intermediate.txt",
    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing dataset names.",
)
args = parser.parse_args()

MODELS_CONFIG = args.models_config

ds_info = pd.read_json("./scripts/configs/webdataset_configs/dataset_info.json").T

if __name__ == "__main__":
    models, n_models = load_models(MODELS_CONFIG)

    model_keys = list(models.keys())

    parsed_datasets = parse_datasets(args.datasets)

    # Extract features for all models and datasets.
    for key in model_keys:
        # for key in ["OpenCLIP_ViT-B-32_openai_cls", "OpenCLIP_ViT-B-32_openai_ap"]:
        # for key in ["mae-vit-base-p16_cls", "mae-vit-base-p16_ap", "mae-vit-base-p16_at", "mae-vit-large-p16_cls", "mae-vit-large-p16_ap", "mae-vit-large-p16_at"]:
        set_length = models[key]["set_length"]
        dim = models[key]["embedding_dim"]

        for dataset in parsed_datasets:
            if dataset in ["imagenet-subset-50k"]:
                print(f"\nSkipping {dataset} because it is the ImageNet1k subset dataset.\n")
                continue
            if dataset == "wds/imagenet1k" and key.endswith("_at"):
                print(
                    f"\nSkipping {dataset} because it is the ImageNet1k dataset and the model is an attention model.\n"
                )
                continue

            curr_ds_info = ds_info.loc[dataset]

            nr_train_samples = curr_ds_info["nr_train_samples"]
            partition = get_partition_based_on_train_samples(nr_train_samples)

            mem = compute_memory_requirements(set_length, nr_train_samples, dim)

            print(f"\nRunning feature extraction for {key} and datasets ({mem}GB memory):\n{dataset}\n")

            job_cmd = f"""python src/cli.py \
                            --dataset {dataset} \
                            --dataset_root {Path(DATASETS_ROOT).absolute()} \
                            --output_root {Path(RESULTS_ROOT).absolute()} \
                            --feature_root {Path(FEATURES_ROOT).absolute()} \
                            --task=feature_extraction \
                            --model_key {key} \
                            --models_config_file {Path(MODELS_CONFIG).absolute()} \
                            --batch_size=128 \
                            --train_split train \
                            --test_split test \
                            --num_workers=0
            """

            run_job(
                job_name=f"feat_extr_{key}",
                job_cmd=job_cmd,
                partition=partition,
                log_dir=f"{FEATURES_ROOT}/logs",
                num_jobs_in_array=1,
                mem=mem,
            )
