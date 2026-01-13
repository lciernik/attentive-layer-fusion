import argparse
import json
from pathlib import Path

from helper import load_models, parse_datasets
from project_location import DATASETS_ROOT, FEATURES_ROOT, MODEL_SIM_ROOT, SUBSET_ROOT
from slurm import run_job

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models_config",
    type=str,
    default="./scripts/configs/models_config_single_model_layer_combination.json",
)
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default="./scripts/configs/webdataset_configs/webdatasets_attentive_probe.txt",
    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing "
    "dataset names.",
)
parser.add_argument("--combine_cls_ap", action="store_true", help="combine cls and ap layers for large sim matrix.")

args = parser.parse_args()

MODELS_CONFIG = args.models_config
DATASETS = parse_datasets(args.datasets)

SIM_METRIC_CONFIG = "./scripts/configs/cka_metrics.json"

with Path(SIM_METRIC_CONFIG).open("r") as file:
    sim_method_config = json.load(file)

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)

    # new_datasets = sorted(DATASETS + ['wds/vtab/pcam'])
    new_datasets = sorted(DATASETS)

    num_jobs = len(new_datasets)

    datasets = " ".join(new_datasets)

    for exp_dict in sim_method_config:
        num_workers = 8

        partition = "cpu-2d"
        mem = 150

        processed = set()
        for model_key, model_config in models.items():

            if "mae" not in model_key:
                print(f"Skipping {model_key} because it is not a MAE model.")
                continue

            if "_at" in model_key:
                print(f"Skipping {model_key} because it is a key indicating to use all tokens of the last layer.")
                continue

            if not ("-B-16" in model_key or "-base-" in model_key or "_base_" in model_key):
                print(f"Skipping {model_key} because it is not a base sized model.")
                continue

            # Skip large vit models if specified
            if args.combine_cls_ap and model_key.replace("_ap", "").replace("_cls", "") in processed:
                print(f"Skipping {model_key} because it has already been processed.")
                continue


            curr_model_keys = " ".join(
                [f"{model_key}@{module}" for module in model_config["module_names"]]
            )
            if args.combine_cls_ap:
                processed.add(model_key.replace("_ap", "").replace("_cls", ""))
                if "_ap" in model_key:
                    curr_model_keys = curr_model_keys + " " + curr_model_keys.replace("_ap", "_cls")
                elif "_cls" in model_key:
                    curr_model_keys = curr_model_keys + " " + curr_model_keys.replace("_cls", "_ap")
                else:
                    raise ValueError(f"{model_key} is not a valid model key.")

            print(
                f">> Computing model similarity matrix for model(s)\n{curr_model_keys}\n"
                f"with config:\n{json.dumps(exp_dict, indent=4)}"
            )

            job_cmd = f"""python src/cli.py \
                        --dataset {datasets} \
                        --dataset_root {Path(DATASETS_ROOT).absolute()} \
                        --feature_root {Path(FEATURES_ROOT).absolute()} \
                        --output {Path(MODEL_SIM_ROOT).absolute()} \
                        --task=model_similarity \
                        --model_key {curr_model_keys} \
                        --models_config_file {Path(MODELS_CONFIG).absolute()} \
                        --train_split train \
                        --sim_method {exp_dict["sim_method"]} \
                        --sim_kernel {exp_dict["sim_kernel"]} \
                        --rsa_method {exp_dict["rsa_method"]} \
                        --corr_method {exp_dict["corr_method"]} \
                        --sigma {exp_dict["sigma"]} \
                        --num_workers {num_workers} \
                        --use_ds_subset \
                        --subset_root {Path(SUBSET_ROOT).absolute()}
                            """

            run_job(
                job_name=f"{exp_dict['sim_method'].capitalize()}",
                job_cmd=job_cmd,
                partition=partition,
                log_dir=f"{MODEL_SIM_ROOT}/logs",
                num_jobs_in_array=num_jobs,
                mem=mem,
            )
