import argparse
import json
import os
import re

from src.utils.loss_utils import Regularization
from src.utils.tasks import Task
from src.utils.utils import as_list

def get_parser_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Get the parser arguments."""
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs) -> None:
        parser.add_argument(*args, **kwargs)

    # DATASET
    aa(
        "--dataset",
        type=str,
        default="cifar10",
        nargs="+",
        help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection "
        "name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file "
        "where each line is a dataset name",
    )
    aa(
        "--pretraining_dataset",
        type=str,
        default="",
        help="Dataset that was used to pretrain a model, e.g., MVAE model.",
    )
    aa(
        "--dataset_root",
        default="root",
        type=str,
        help="dataset root folder where the data are downloaded. Can be in the form of a "
        "template depending on dataset name, e.g., --dataset_root='data/{dataset}'. "
        "This is useful if you evaluate on multiple data.",
    )
    aa("--split", type=str, default="test", help="Dataset split to use")
    aa(
        "--test_split",
        dest="split",
        action="store",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    aa("--train_split", type=str, default="train", help="Dataset(s) train split names")
    aa(
        "--val_proportion",
        default=0.2,
        type=float,
        help="what is the share of the train dataset will be used for validation part, if it doesn't predefined.",
    )
    aa(
        "--wds_cache_dir",
        default=None,
        type=str,
        help="optional cache directory for webdataset only",
    )

    # FEATURES
    aa(
        "--feature_root",
        default="features",
        type=str,
        help="feature root folder where the features are stored.",
    )
    aa(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="enable features normalization",
    )
    aa(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="disable features normalization",
    )

    # MODEL(S)
    aa(
        "--model_key",
        type=str,
        nargs="+",
        default=["dinov2-vit-large-p14"],
        help="""
        Models to use from the models config file. When task=rep2rep, order is important!
        We learn to transfer from the first model to the second model.
        """,
    )
    aa(
        "--models_config_file",
        default=None,
        type=str,
        help="Path to the models config file.",
    )

    # TASKS
    aa(
        "--task",
        type=Task,
        default=Task.LINEAR_PROBE,
        choices=list(Task),
        help="Task to evaluate.",
    )
    aa(
        "--mode",
        type=str,
        default="single_model",
        choices=["single_model", "combined_models", "ensemble", "linear", "mvae", "mvae_eval", "end_2_end"],
        help="Modes for linear and attentive probe tasks, as well as for the rep2rep task. Linear probe modes: single_model, combined_models, ensemble, mvae_eval (evaluate MVAE embeddings), end_2_end (uses image dataloader instead of preextracted features). Attentive probe modes: combined_models or end_2_end (uses image dataloader instead of preextracted features). Rep2rep modes: linear, mvae.",
    )
    aa(
        "--feature_combiner",
        type=str,
        default="concat",
        choices=["concat", "concat_pca", "tuple", "stacked_zero_pad", "already_stacked_zero_pad"],
        help="Feature combiner to use. Used only for modes: combined_models, mvae, mvae_eval. The attentive probe task or the MVAE task, can only be used with the tuple feature combiner."
        "For End2End, only 'concat' or 'stacked_zero_pad' is allowed and prepares the output of the premodel in preparation for the classification head.",
    )

    aa(
        "--freeze_premodel",
        default=True,
        action="store_true",
        help="Freeze the premodel during end_2_end training.",
    )
    aa(
        "--unfreeze_premodel",
        dest="freeze_premodel",
        action="store_false",
        help="Unfreeze the premodel during end_2_end training.",
    )

    # LINEAR PROBE PARAMETERS
    aa(
        "--fewshot_k",
        default=[-1],
        type=int,
        nargs="+",
        help="Number of examples per class for few-shot learning. Use -1 for the entire dataset.",
    )
    aa(
        "--epochs",
        default=[10],
        type=int,
        nargs="+",
        help="Number of training epochs for the few-shot learning model.",
    )
    aa(
        "--initial_lr",
        default=[0.1],
        type=float,
        nargs="+",
        help="Learning rate for training the few-shot learning model. If val_proportion > 0, the best learning rate "
        "will be found from the list provided. If val_proportion == 0, the list should contain exactly one value, "
        "which will be used for training.",
    )
    aa(
        "--regularization",
        default=["weight_decay"],
        type=str,
        nargs="+",
        help="Type of regularization applied during training.",
        choices=Regularization.values(),
    )
    aa(
        "--reg_lambda",
        default=0.2,
        type=float,
        help="Regularization parameter (lambda, weight decay value) for training the model. "
        "This parameter is used only when val_proportion==0. If val_proportion>0, the optimal value will be "
        "determined through a search within a fixed range.",
    )
    aa(
        "--reg_lambda_bounds",
        default= (-6, 0),
        type=int,
        nargs=2,
        help="Exponent bounds for searching the optimal regularization parameter "
        "when val_proportion>0. The search will be performed in the range [10^min_exp, 10^max_exp].",
    )

    aa(
        "--grad_norm_clip",
        default=5,
        type=float,
        help="Gradient norm clipping value.",
    )
    aa(
        "--jitter_p",
        type=float,
        default=0.5,
        help="Probability of applying jitter to the features. By default it uses N(mu=0, sigma=0.05).",
    )
    aa("--batch_size", default=64, type=int, help="Batch size for training.")
    aa(
        "--skip_existing",
        default=False,
        action="store_true",
        help="Skip the evaluation if the output file already exists.",
    )
    aa(
        "--force_train",
        default=False,
        action="store_true",
        help="Retrain linear probe even if model already exists.",
    )
    aa(
        "--no_class_weights",
        dest="use_class_weights",
        action="store_false",
        help="Do NOT use class weights for training the evaluation probe. By default, class weights are used.",
    )

    aa(
        "--rep_loss",
        type=rep_loss_type,
        default="mse",
        help="Loss function for rep2rep task. Options:"
        " (1) Single losses: 'mse', 'mae', 'cosine_distance', 'glocal_TEMPT' or 'glocal_TEMPT_TEMPS'"
        " (where TEMPT/TEMPS are floats - single value sets same temperature for both teacher and student,"
        " two values set teacher to first and student to second),"
        " 'cka_linear' or 'cka_rbf_SIGMA' (where SIGMA is a float);"
        " (2) Combined losses: 'combinedALPHA__L1__L2'"
        " where ALPHA (between 0 and 1) determines the weight as ALPHA*L1 + (1-ALPHA)*L2.",
    )
    ## TODO: add these parameters to tasks other than SAE
    aa(
        "--final_lr",
        default=1e-6,
        type=float,
        help="Final learning rate for training the SAE.",
    )
    aa(
        "--warmup_epochs",
        default=5,
        type=int,
        help="Number of warmup epochs for the SAE.",
    )
    aa(
        "--grad_clip",
        default=5,
        type=float,
        help="Gradient clipping value for the SAE.",
    )
    aa(
        "--patience",
        type=int,
        default=10,
        help="Number of patience for the SAE.",
    )
    aa(
        "--min_delta",
        type=float,
        default=1e-5,
        help="Minimum delta for the SAE.",
    )
    aa(
        "--extract_train",
        default=False,
        action="store_true",
        help="Extract train latents for the SAE.",
    )
    aa(
        "--extract_test",
        default=False,
        action="store_true",
        help="Extract test latents for the SAE.",
    )
    aa(
        "--lin_probe_eval",
        action="store_true",
        default=True,
        help="Evaluate the hidden representations with a linear probe.",
    )
    aa(
        "--no_lin_probe_eval",
        dest="lin_probe_eval",
        action="store_false",
        help="Disable linear probe evaluation of hidden representations.",
    )
    aa(
        "--dim",
        type=int,
        default=None,
        help="Shared dimension of the attentive probe.",
    )
    aa(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads for the attentive probe.",
    )
    aa(
        "--attention_dropout",
        type=float,
        nargs=2,  # require exactly two floats
        default=(0.0, 0.0),
        help="Dropout rates for attention: (proj_drop, attn_drop).",
    )

    aa(
        "--dimension_alignment",
        type=str,
        default="zero_padding",
        choices=["zero_padding", "linear_projection"],
        help="Dimension alignment method for the attentive probe.",
    )
    aa(
        "--project_dims_on_mismatch",
        action="store_true",
        help="Only project the input features to the shared dimension if the shared dimension is not equal to the dimension of the input features. (Default: always project)",
    )
    aa(
        "--num_clusters",
        type=int,
        default=-1,
        help="Number of clusters to use for the combined model evaluation (linear or attentive probe)."
        " Models/ Layers of the same model will be clustered based on their representational similarity,"
        " and only one representative model from each cluster will be used by the probe."
        " If -1, clustering is disabled and all models are used.",
    )
    aa(
        "--clustering_similarity_method",
        type=str,
        default="cka_kernel_linear_unbiased",
        help="Method to use for model similarity task during clustering.",
    )
    # SAE TRAINING PARAMETERS
    aa(
        "--sae_k",
        type=int,
        default=None,
        help="Number of latent features for the SAE. If not provided, the number of latent features will be "
        "determined automatically.",
    )
    aa(
        "--sae_increase_factor",
        type=int,
        default=8,
        help="Increase factor for the number of latent features for the SAE.",
    )

    # WANDB
    aa(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Use wandb to log the SAE training.",
    )

    ### Model similarity
    aa(
        "--sim_method",
        type=str,
        default="cka",
        choices=["cka", "rsa", "cosine"],
        help="Method to use for model similarity task.",
    )
    aa(
        "--sim_kernel",
        type=str,
        default="linear",
        choices=["linear", "rbf"],
        help="Kernel used during CKA. Ignored if sim_method is rsa.",
    )
    aa(
        "--rsa_method",
        type=str,
        default="correlation",
        choices=["cosine", "correlation"],
        help="Method used during RSA. Ignored if sim_method is cka.",
    )
    aa(
        "--corr_method",
        type=str,
        default="spearman",
        choices=["pearson", "spearman"],
        help="Kernel used during CKA. Ignored if sim_method is cka.",
    )
    aa("--sigma", type=float, default=None, help="sigma for CKA rbf kernel.")
    aa("--biased_cka", action="store_false", dest="unbiased", help="use biased CKA")
    aa(
        "--use_ds_subset",
        action="store_true",
        help="Compute model similarities on precomputed subset of the dataset.",
    )
    aa(
        "--subset_root",
        type=str,
        help="Path to the root folder where the dataset subset indices are stored. "
        "Only used if use_ds_subset is True.",
    )

    # STORAGE
    aa(
        "--output_root",
        default="results",
        type=str,
        help="Path to root folder where the results are stored.",
    )
    aa(
        "--model_root",
        default="models",
        type=str,
        help="Path to root folder where linear probe model checkpoints are stored.",
    )
    aa(
        "--clustering_root",
        default="clustering",
        type=str,
        help="Path to root folder where clustering results are stored.",
    )
    # GENERAL
    aa("--num_workers", default=0, type=int)

    aa("--distributed", action="store_true", help="evaluation in parallel")
    aa(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="suppress verbose messages",
    )

    # REPRODUCABILITY
    aa("--seed", default=[0], type=int, nargs="+", help="random seed.")

    args = parser.parse_args()

    args.always_project = not args.project_dims_on_mismatch

    if args.pretraining_dataset in ["FALSE", "False", "false", "f", "F"]:
        args.pretraining_dataset = ""

    ## INPUT VALIDATION
    if args.num_workers > 0 and args.task == Task.FEATURE_EXTRACTION:
        raise ValueError("At the moment we only allow for num_workers=0.")

    if args.task == Task.ATTENTIVE_PROBE:
        if args.mode == "combined_models":
            if args.feature_combiner not in ["tuple", "stacked_zero_pad", "already_stacked_zero_pad"]:
                raise ValueError(
                    "At the moment we only allow  'tuple', 'stacked_zero_pad', and 'already_stacked_zero_pad' feature combiners for the combined_models mode in the attentive probe task."
                )
        elif args.mode == "end_2_end":
            if args.feature_combiner not in ["stacked_zero_pad"]:
                raise ValueError(
                    "At the moment we only allow 'stacked_zero_pad' activation combiners for the end_2_end mode in the attentive probe task."
                )
        else:
            raise ValueError(
                "At the moment we only allow for combined_models and end_2_end mode for attentive probe task."
            )
    elif args.task == Task.LINEAR_PROBE:
        if args.mode == "combined_models" and args.feature_combiner not in ["concat", "concat_pca"]:
            raise ValueError(
                "At the moment we only allow for linear probe with combined_models mode a "
                "concat or concat_pca feature combiner."
            )
        elif args.mode == "end_2_end" and args.feature_combiner != "concat":
            raise ValueError(
                "At the moment we only allow for linear probe with end_2_end mode a concat feature_combiner combiner."
            )

    if args.grad_norm_clip is not None and not isinstance(args.grad_norm_clip, (int, float)):
        raise ValueError("Gradient norm clipping value must be a number.")

    if args.jitter_p < 0 or args.jitter_p > 1:
        raise ValueError("Jitter probability must be between 0 and 1.")

    return parser, args


def prepare_args(args: argparse.Namespace, model_info: tuple[str, str, dict, str, str, str]) -> argparse.Namespace:
    """Prepare the arguments for the model."""
    args.model = model_info[0]  # model
    args.model_source = model_info[1]  # model_source
    args.model_parameters = model_info[2]  # model_parameters
    args.module_names = model_info[3]  # module_names
    args.feature_alignment = model_info[4]  # feature_alignment
    args.model_key = model_info[5]  # model_key
    return args


def prepare_combined_args(
    args: argparse.Namespace, model_comb: list[tuple[str, str, dict, str, str, str]]
) -> argparse.Namespace:
    """Prepare the arguments for the model combination."""
    args.model = [tup[0] for tup in model_comb]
    args.model_source = [tup[1] for tup in model_comb]
    args.model_parameters = [tup[2] for tup in model_comb]
    args.module_names = [tup[3] for tup in model_comb]
    args.feature_alignment = [tup[4] for tup in model_comb]
    args.model_key = [tup[5] for tup in model_comb]
    return args


def load_model_configs_args(base: argparse.Namespace) -> argparse.Namespace:
    """Loads the model_configs file and transcribes its parameters into base."""
    if base.models_config_file is None:
        raise FileNotFoundError("Model config file not provided.")

    if not os.path.exists(base.models_config_file):
        raise FileNotFoundError(f"Model config file {base.models_config_file} does not exist.")

    with open(base.models_config_file, "r") as f:
        model_configs = json.load(f)

    model = []
    model_source = []
    model_parameters = []
    module_names = []
    feature_alignment = []

    for model_key in as_list(base.model_key):
        ## For intermediate layer extraction, we need to extract the model name from the model_key
        if "@" in model_key:
            model_key, current_module_name = model_key.split("@")
        else:
            current_module_name = model_configs[model_key]["module_names"]

        model.append(model_configs[model_key]["model_name"])
        model_source.append(model_configs[model_key]["source"])
        model_parameters.append(model_configs[model_key]["model_parameters"])
        module_names.append(current_module_name)
        feature_alignment.append(model_configs[model_key]["alignment"])

    setattr(base, "model", model)
    setattr(base, "model_source", model_source)
    setattr(base, "model_parameters", model_parameters)
    setattr(base, "module_names", module_names)
    setattr(base, "feature_alignment", feature_alignment)

    return base


def rep_loss_type(value):
    # Predefined choices
    predefined_choices = {"mse", "cosine_distance", "mae", "cka_linear"}

    # Allow dynamic cka_rbf_FLOAT values
    if value in predefined_choices:
        return value
    elif re.match(
        r"^(mvaedreg|mvaeelbo)(?:K\d+)?(scale\d*(?:\.\d*)?)?(?:NL)?(?:normalize)?(?:IndVariance)?_(poe|moe)_(loose|strict)_(laplace|normal)$",
        value,
    ):
        return value
    elif re.match(r"^combined(\d+(?:\.\d+)?)(?:__(.+))?$", value):
        return value
    elif re.match(r"^glocal_(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$", value):  # Regex for glocal_FLOAT
        return value
    elif re.match(r"^cka_rbf_\d+(\.\d+)?$", value):  # Regex for cka_rbf_FLOAT
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid choice: {value}. Must be one of {predefined_choices} or match 'cka_rbf_FLOAT'."
        )
