import os
import sys
from pathlib import Path

sys.path.append("..")

from scripts.project_location import BASE_PATH_PROJECT as bpp
from scripts.project_location import RESULTS_ROOT as bpr
from scripts.project_location import SUBSTRING

########################################################################
## DEFINE BASEPATHS
BASE_PATH_PROJECT = Path(bpp)
BASE_PATH_RESULTS = Path(bpr)
try:
    REPOSITORY_PATH = Path(os.environ.get("PATH_TO_REPO"))
except:
    REPOSITORY_PATH = Path("/srv/repo")

FOLDER_SUBSTRING = SUBSTRING

########################################################################
## Path to the model config file
model_config_file = REPOSITORY_PATH / "scripts" / "configs" / "models_config_single_model_layer_combination.json"
ds_info_file = REPOSITORY_PATH / "scripts" / "configs" / "webdataset_configs" / "dataset_info.json"
ds_list_file = REPOSITORY_PATH / "scripts" / "configs" / "webdataset_configs" / "webdatasets_part_one_experiments.txt"

########################################################################
## DEFINE CONSTANT LISTS

similarity_metrics = [
    "cka_kernel_rbf_unbiased_sigma_0.2",
    "cka_kernel_rbf_unbiased_sigma_0.4",
    "cka_kernel_rbf_unbiased_sigma_0.6",
    "cka_kernel_rbf_unbiased_sigma_0.8",
    "cka_kernel_linear_unbiased",
    "rsa_method_correlation_corr_method_pearson",
    "rsa_method_correlation_corr_method_spearman",
]


experiment_with_probe_type_order_list = [
    "CLS last layer",  # baseline 1
    "AP last layer",  # baseline 1
    "All tokens last layer (attentive)",  # baseline 2
    "CLS+AP last layer (linear)",
    "CLS+AP layers from middle & last blocks (linear)",
    "CLS+AP layers from quarterly blocks (linear)",
    "CLS+AP layers from all blocks (linear)",
    "CLS+AP last layer (attentive)",
    "CLS+AP layers from middle & last blocks (attentive)",
    "CLS+AP layers from quarterly blocks (attentive)",
    "CLS+AP layers from all blocks (attentive)",
]

experiment_order_list = [
    "CLS last layer",  # baseline 1
    "All tokens last layer",  # baseline 2
    "CLS+AP last layer",
    "CLS+AP layers from middle & last blocks",
    "CLS+AP layers from quarterly blocks",
    "CLS+AP layers from all blocks",
]

DS_ORDER = [
    # Natural (MD)
    "STL-10",
    "CIFAR-10",
    "Caltech-101",
    "PASCAL VOC 2007",
    "CIFAR-100",
    "Country-211",
    # Natural (SD)
    "Pets",
    "Flowers",
    "Stanford Cars",
    "FGVC Aircraft",
    "GTSRB",
    "SVHN",
    # Specialized "PCAM",
    "PCAM",
    "EuroSAT",
    "RESISC45",
    "Diabetic Retinopathy",
    # Structured
    "DTD",
    "FER2013",
    "Dmlab",
]

########################################################################
## DEFINE NAME MAPPINGS
sim_metric_name_mapping = {
    "cka_kernel_rbf_unbiased_sigma_0.2": "CKA RBF 0.2",
    "cka_kernel_rbf_unbiased_sigma_0.4": "CKA RBF 0.4",
    "cka_kernel_rbf_unbiased_sigma_0.6": "CKA RBF 0.6",
    "cka_kernel_rbf_unbiased_sigma_0.8": "CKA RBF 0.8",
    "cka_kernel_linear_unbiased": "CKA linear",
    "rsa_method_correlation_corr_method_pearson": "RSA pearson",
    "rsa_method_correlation_corr_method_spearman": "RSA spearman",
}


base_model_name_mapping = {
    "OpenCLIP_ViT-B-16_openai": "CLIP-B-16",
    "OpenCLIP_ViT-B-32_openai": "CLIP-B-32",
    "OpenCLIP_ViT-L-14_openai": "CLIP-L-14",
    "dinov2-vit-small-p14": "DINOv2-S-14",
    "dinov2-vit-base-p14": "DINOv2-B-14",
    "dinov2-vit-large-p14": "DINOv2-L-14",
    "vit_small_patch16_224": "ViT-S-16",
    "vit_base_patch16_224": "ViT-B-16",
    "vit_large_patch16_224": "ViT-L-16",
    "mae-vit-base-p16": "MAE-B-16",
    "mae-vit-large-p16": "MAE-L-16",
}

########################################################################
## PLOTTING CONSTANTS
fontsizes = {
    "title": 14,
    "legend": 13,
    "label": 13,
    "ticks": 12,
}

fontsizes_cols = {
    "title": 18,
    "legend": 17,
    "label": 17,
    "ticks": 16,
}

cm = 0.393701

DS_ORDER = [
    "STL-10", "CIFAR-10", "Caltech-101", "PASCAL VOC 2007", "CIFAR-100", "Country-211",
    "Pets", "Flowers", "Stanford Cars", "FGVC Aircraft", "GTSRB", "SVHN",
    "PCAM", "EuroSAT", "RESISC45", "Diabetic Retinopathy",
    "DTD", "FER2013", "Dmlab"
]
