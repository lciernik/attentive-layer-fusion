import argparse
from pathlib import Path

from scripts.helper import get_hyperparams, load_models, parse_datasets
from scripts.project_location import DATASETS_ROOT, FEATURES_ROOT, MODELS_ROOT, RESULTS_ROOT
from scripts.slurm import run_job

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models_config", type=str, default="./scripts/configs/models_config_single_model_layer_combination.json"
)
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default="./scripts/configs/webdataset_configs/webdatasets_ext_experiments.txt",
    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing dataset names.",
)

args = parser.parse_args()

MODELS_CONFIG = args.models_config
DATASETS = " ".join(parse_datasets(args.datasets))

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)

    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=1, size="imagenet1k")

    # With val_proportion 0 we do not optimize weight decay!
    val_proportion = 0.2

    # Evaluate
    for i, (mid, m_cfg) in enumerate(models.items()):
        if "_cls" not in mid:
            continue
        m_cfg = models[mid]

        # Take each model from the config and evaluate last layer
        key = f"{mid}@{m_cfg['module_names'][-1]}"

        print(f"Running single model linear probe evaluation for {key}")

        job_cmd = f"""python src/cli.py \
                        --dataset {DATASETS} \
                        --dataset_root {Path(DATASETS_ROOT).absolute()} \
                        --feature_root {Path(FEATURES_ROOT).absolute()} \
                        --model_root {Path(MODELS_ROOT).absolute()} \
                        --output_root {Path(RESULTS_ROOT).absolute()} \
                        --task=linear_probe \
                        --mode=single_model \
                        --model_key {key} \
                        --models_config_file {Path(MODELS_CONFIG).absolute()} \
                        --batch_size=2048 \
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
                        --jitter_p 0.5
        """

        run_job(
            job_name=f"probe_{key}",
            job_cmd=job_cmd,
            partition="gpu-2d",
            log_dir=f"{Path(RESULTS_ROOT).absolute()}/logs",
            num_jobs_in_array=num_jobs,
            mem=100,
        )
