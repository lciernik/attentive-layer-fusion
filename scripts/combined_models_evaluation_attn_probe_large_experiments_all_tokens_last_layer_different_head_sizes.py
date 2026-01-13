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
    nargs="+",
    type=str,
    default=[
        "./scripts/configs/all_tokens_one_layer/model_combinations_all_tokens_last_layer_B.txt",
    ],
    help="File containing model combinations to evaluate with an attentive probe.",
)

args = parser.parse_args()

MODELS_CONFIG = args.models_config


## Best parameters with 8 heads:
ATTENTION_DROPOUT = {
    "OpenCLIP_ViT-B-16_openai_at": {
        "wds/fer2013": [0.0, 0.3],
        "wds/gtsrb": [0.0, 0.3],
        "wds/vtab/cifar100": [0.0, 0.1],
        "wds/vtab/eurosat": [0.0, 0.1],
    },
    "dinov2-vit-base-p14_at": {
        "wds/fer2013": [0.0, 0.3],
        "wds/gtsrb": [0.0, 0.0],
        "wds/vtab/cifar100": [0.0, 0.0],
        "wds/vtab/eurosat": [0.0, 0.0],
    },
    "vit_base_patch16_224_at": {
        "wds/fer2013": [0.0, 0.3],
        "wds/gtsrb": [0.0, 0.0],
        "wds/vtab/cifar100": [0.0, 0.1],
        "wds/vtab/eurosat": [0.0, 0.0],
    },
}

LEARNING_RATES = {
    "OpenCLIP_ViT-B-16_openai_at": {
        "wds/fer2013": 0.1,
        "wds/gtsrb": 0.001,
        "wds/vtab/cifar100": 0.001,
        "wds/vtab/eurosat": 0.1,
    },
    "dinov2-vit-base-p14_at": {
        "wds/fer2013": 0.001,
        "wds/gtsrb": 0.001,
        "wds/vtab/cifar100": 0.001,
        "wds/vtab/eurosat": 0.001,
    },
    "vit_base_patch16_224_at": {
        "wds/fer2013": 0.001,
        "wds/gtsrb": 0.1,
        "wds/vtab/cifar100": 0.001,
        "wds/vtab/eurosat": 0.001,
    },
}


if __name__ == "__main__":
    if not isinstance(args.models_combination, list):
        args.models_combination = [args.models_combination]

    prepared_datasets = sorted(set(parse_datasets(args.datasets)))

    ds_to_remove = ["wds/imagenet1k", "imagenet-subset-50k"]  # , "wds/vtab/pcam"]
    prepared_datasets = sorted(set(prepared_datasets) - set(ds_to_remove))

    for file_to_model_combinations in args.models_combination:
        # Parse the model comgsbinations from the file
        if os.path.isfile(file_to_model_combinations):
            with open(file_to_model_combinations, "r") as f:
                model_combinations = [[m.strip() for m in line.split(";")] for line in f if line.strip()]
        else:
            raise ValueError("The file does not exist", file_to_model_combinations)

        # Load the models configuration and set global variables
        models, n_models = load_models(MODELS_CONFIG)

        hyper_params, num_jobs = get_hyperparams(num_seeds=1, size="imagenet1k-small")

        val_proportion = 0

        dim_alignment = "zero_padding"

        for DATASET in prepared_datasets:
            for model_set in model_combinations:
                for n_heads in [12, 16, 32]:
                    assert all([key.split("@")[0] in models.keys() for key in model_set])

                    model_keys = " ".join(model_set)

                    if "_at@" not in model_set[0]:
                        continue

                    first_model = model_set[0].split("@")[0]
                    dim = models[first_model]["embedding_dim"]
                    nr_models = models[first_model]["set_length"]
                    feature_combiner = "already_stacked_zero_pad"

                    if not (first_model == "dinov2-vit-base-p14_at" and DATASET == "wds/gtsrb"):
                        continue

                    nr_models = min(nr_models, 50)
                    mem = int(50 + (nr_models * 10))

                    initial_lr = LEARNING_RATES[first_model][DATASET]
                    proj_drop = ATTENTION_DROPOUT[first_model][DATASET][0]
                    att_drop = ATTENTION_DROPOUT[first_model][DATASET][1]

                    print(
                        f"\nRunning attentive probe for:\n"
                        f"{num_jobs=} jobs,\n"  
                        f"{model_set=},\n"  
                        f"{proj_drop=} dropout,\n"
                        f"{att_drop=} dropout,\n"
                        f"{n_heads=} heads,\n"
                        f"{DATASET=} dataset,\n"
                        f"{initial_lr=} initial learning rate,\n"
                        f"and {mem}GB memory\n"
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
                                    --initial_lr {initial_lr} \
                                    --epochs {" ".join(hyper_params["epochs"])} \
                                    --reg_lambda 0.1 \
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
                        # partition="gpu-2d",
                        partition="gpu-5h",
                        log_dir=f"{Path(RESULTS_ROOT).absolute()}/logs",
                        num_jobs_in_array=num_jobs,
                        mem=mem,
                    )
