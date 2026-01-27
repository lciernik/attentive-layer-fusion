import os

### DEFINE PROJECT IMAGE
BASE_PROJECT_IMAGE = "oras://ghcr.io/lciernik/attentive-layer-fusion:latest-sif"
ADDITIONAL_MOUNTS_w_copy = [
    "/home/space/diverse_priors",
    "/tmp/imagenet_torchvision.sqfs:/imagenet_torchvision:image-src=/",
]
ADDITIONAL_MOUNTS_wo_copy = [
    "/home/space/diverse_priors",
    "/home/space/datasets-sqfs/imagenet_torchvision.sqfs:/imagenet_torchvision:image-src=/",
]

## DEFINE BASEPATHS
BASE_PATH_PROJECT = ".." # TODO: set appropriate path
SUBSTRING = ".." # TODO: set appropriate string

### DEFINE SUBFOLDERS

DATASETS_ROOT = os.path.join(BASE_PATH_PROJECT, "datasets")

SUBSET_ROOT = os.path.join(DATASETS_ROOT, "subsets")

FEATURES_ROOT = os.path.join(BASE_PATH_PROJECT, "features")

# MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, "models")
# MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, f"models_{SUBSTRING}")
# MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, f"models_{SUBSTRING}_exp_cls_vs_ap")
# MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, f"models_{SUBSTRING}_rebuttal")
# MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, f"models_{SUBSTRING}_combine_aat_ilf")
MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, f"models_{SUBSTRING}_aat_diff_nr_heads")


MODEL_SIM_ROOT = os.path.join(BASE_PATH_PROJECT, "model_similarities_{SUBSTRING}_exp")

# RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, "results")
# RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, f"results_{SUBSTRING}_exp")
# RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, f"results_{SUBSTRING}_exp_cls_vs_ap")
# RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, f"results_{SUBSTRING}_rebuttal")
# RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, f"results_{SUBSTRING}_combine_aat_ilf")
RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, f"results_{SUBSTRING}_aat_diff_nr_heads")

CLUSTERING_ROOT = os.path.join(BASE_PATH_PROJECT, "clustering")


if __name__ == "__main__":
    paths = [
        DATASETS_ROOT,
        SUBSET_ROOT,
        FEATURES_ROOT,
        MODELS_ROOT,
        MODEL_SIM_ROOT,
        RESULTS_ROOT,
    ]
    if not BASE_PATH_PROJECT:
        raise ValueError(
            "Please set the BASE_PATH_PROJECT variable in project_location.py to the project folder path."
        )

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")
