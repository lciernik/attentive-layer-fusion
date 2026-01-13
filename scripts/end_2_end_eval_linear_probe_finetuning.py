import argparse
import os
from pathlib import Path

from scripts.helper import get_hyperparams, load_models, parse_datasets
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
    default="./scripts/configs/webdataset_configs/webdatasets_all_intermediate.txt",
    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing dataset names.",
)

parser.add_argument(
    "--models_combination",
    type=str,
    default="./scripts/configs/finetuning/model_combinations_fine_tuning_B.txt",
    help="File containing model combinations to evaluate with a linear probe.",
)

args = parser.parse_args()

MODELS_CONFIG = args.models_config
DATASETS = parse_datasets(args.datasets)

if __name__ == "__main__":
    if os.path.isfile(args.models_combination):
        with open(args.models_combination, "r") as f:
            model_combinations = [[m.strip() for m in line.split(";")] for line in f if line.strip()]
    else:
        raise ValueError("The file does not exist", args.models_combination)

    models, n_models = load_models(MODELS_CONFIG)

    hyper_params, num_jobs = get_hyperparams(num_seeds=1, size="small_finetuning")

    val_proportion = 0

    for model_set in model_combinations:
        assert all([key.split("@")[0] in models.keys() for key in model_set])
        print(f"Running finetuning for {model_set}")
        model_keys = " ".join(model_set)

        for dataset in DATASETS:
            print(f">> Running finetuning for {model_set} on {dataset}")

            job_cmd = f"""python src/cli.py --dataset {dataset} \
                            --dataset_root {Path(DATASETS_ROOT).absolute()} \
                            --feature_root {Path(FEATURES_ROOT).absolute()} \
                            --model_root {Path(MODELS_ROOT).absolute()} \
                            --output_root {Path(RESULTS_ROOT).absolute()} \
                            --clustering_root {Path(CLUSTERING_ROOT).absolute()} \
                            --task=linear_probe \
                            --mode=end_2_end \
                            --feature_combiner concat \
                            --model_key {model_keys} \
                            --models_config_file {MODELS_CONFIG} \
                            --batch_size=256 \
                            --fewshot_k {" ".join(hyper_params["fewshot_ks"])} \
                            --initial_lr {" ".join(hyper_params["initial_lrs"])} \
                            --epochs {" ".join(hyper_params["epochs"])} \
                            --reg_lambda {hyper_params["reg_lambda"]} \
                            --regularization {" ".join(hyper_params["regularization"])} \
                            --train_split train \
                            --test_split test \
                            --val_proportion {val_proportion} \
                            --seed {" ".join(hyper_params["seeds"])} \
                            --grad_norm_clip 5 \
                            --jitter_p 0.5 \
                            --unfreeze_premodel
            """
            run_job(
                job_name=f"end2end_linear_probe_with_n{len(model_set)}",
                job_cmd=job_cmd,
                partition="gpu-2d",
                log_dir=f"{Path(RESULTS_ROOT).absolute()}/logs",
                num_jobs_in_array=num_jobs,
                mem=40 + int(0.5 * len(model_set)),
                copy_sqfs=False,
                constraint="'[80gb|h100]'"
            )
