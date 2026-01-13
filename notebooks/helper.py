import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.append("..")

from notebooks.constants import base_model_name_mapping, ds_info_file, fontsizes
from scripts.helper import parse_datasets


def plot_r_coeff_distribution(df, sim_met_col, r_x, r_y="gap", ds_col="dataset"):
    r_vals = []
    for key, group_data in df.groupby([ds_col, sim_met_col]):
        r = group_data[r_x].corr(group_data[r_y], method="spearman")
        r_vals.append(
            {
                "Dataset": key[0],
                sim_met_col: key[1],
                "r_coeff": r,
            }
        )

    r_values = pd.DataFrame(r_vals)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    sns.boxplot(
        r_values,
        x=sim_met_col,
        y="r_coeff",
        ax=axs[0],
    )
    sns.histplot(
        r_values,
        x="r_coeff",
        hue=sim_met_col,
        bins=10,
        multiple="dodge",
        kde=True,
        ax=axs[1],
        alpha=0.5,
    )
    sns.kdeplot(r_values, x="r_coeff", hue=sim_met_col, ax=axs[2])

    for i in range(3):
        axs[i].set_xlabel("Correlation coefficient")

    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Distibution correlation coefficients over all datasets.")

    vmin = max(r_values["r_coeff"].min(), -0.5)
    vmax = min(r_values["r_coeff"].max(), 0.5)

    for idx, val in product([1, 2], [vmin, vmax]):
        axs[idx].axvline(val, ls=":", c="grey", alpha=0.5)

    return fig


def plot_scatter(df, title, ds, sim_met_col, sim_val_col):
    g = sns.relplot(
        df,
        x=sim_val_col,
        y="gap",
        col=sim_met_col,
        row="dataset",
        height=3,
        aspect=1.25,
        facet_kws={"sharey": False, "sharex": False},
    )
    g.set_titles("{row_name} – {col_name}")
    g.set_ylabels("Performance gap combined - anchor")
    g.set_xlabels(f"Similarity value {ds}.")

    def annotate_correlation(data, **kwargs):
        r = data[sim_val_col].corr(data["gap"], method="spearman")
        ax = plt.gca()
        ax.text(
            0.05,
            0.95,
            f"r = {r:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        if max(data["gap"]) > 0:
            ax.axhspan(
                0, max(data["gap"]), facecolor="lightgreen", alpha=0.2, zorder=-1
            )
        if min(data["gap"]) < 0:
            ax.axhspan(
                min(data["gap"]), 0, facecolor="lightcoral", alpha=0.2, zorder=-1
            )

    g.map_dataframe(annotate_correlation)

    g.fig.suptitle(title, y=1)
    g.fig.tight_layout()
    return g.fig


def save_or_show(fig, path, save):
    if save == "both":
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"stored image.")
        plt.show(fig)
    elif save:
        fig.savefig(path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"stored img at{ ' ' + str(path) if show_path else ''}.")
    else:
        plt.show(fig)


def get_model_ids(fn: Union[str, Path]) -> List[str]:
    """
    Load model ids from file.
    Args:
        fn: Path to file containing model ids.

    Returns:
        List of model ids.
    """
    with open(fn, "r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


def load_sim_matrix(
    path: Union[str, Path], allowed_models: List[str] | None
) -> pd.DataFrame:
    """
    Load similarity matrix from file and filter for allowed models.
    Args:
        path: Path to similarity matrix.
        allowed_models: List of allowed model ids.

    Returns:

    """
    model_ids_fn = path / "model_ids.txt"
    sim_mat_fn = path / "similarity_matrix.pt"
    if model_ids_fn.exists():
        model_ids = get_model_ids(model_ids_fn)
    else:
        raise FileNotFoundError(f"{str(model_ids_fn)} does not exist.")
    sim_mat = torch.load(sim_mat_fn)
    df = pd.DataFrame(sim_mat, index=model_ids, columns=model_ids)
    if allowed_models is not None:
        df = df.loc[allowed_models, allowed_models]
    return df


def load_similarity_matrices(
    path: Union[str, Path],
    ds_list: List[str],
    sim_metrics: List[str],
    allowed_models: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load similarity matrices for all datasets and similarity metrics.
    Args:
        ds_list: List of dataset names
        sim_metrics: List of similarity metrics
        path: Base path to similarity matrices
        allowed_models: List of allowed model ids

    Returns:
        Dictionary of similarity matrices for all similarity metrics and datasets. Structure: {sim_metric: {ds: pd.DataFrame}}

    """
    sim_mats = defaultdict(dict)
    for sim_metric in sim_metrics:
        for ds in ds_list:
            sim_mats[sim_metric][ds] = load_sim_matrix(
                path / ds / sim_metric, allowed_models
            )
            np.fill_diagonal(sim_mats[sim_metric][ds].values, 1)
    return sim_mats


def load_model_configs_and_allowed_models(
    path: Union[str, Path],
    exclude_models: List[str] = [
        "SegmentAnything_vit_b",
        "DreamSim_dino_vitb16",
        "DreamSim_open_clip_vitb32",
    ],
    exclude_alignment: bool = True,
    sort_by: str = "objective",
) -> Tuple[pd.DataFrame, List[str]]:
    with open(path, "r") as f:
        model_configs = json.load(f)

    print(f"Nr. models original={len(model_configs)}")
    models_to_exclude = [
        k
        for k, v in model_configs.items()
        if (exclude_alignment and v["alignment"] is not None) or (k in exclude_models)
    ]
    if models_to_exclude:
        for k in models_to_exclude:
            model_configs.pop(k)
        print(f"Nr. models after exclusion={len(model_configs)}")

    model_configs = pd.DataFrame(model_configs).T
    model_configs = (
        model_configs.reset_index().sort_values([sort_by, "index"]).set_index("index")
    )

    allowed_models = model_configs.index.tolist()

    return model_configs, allowed_models


def load_ds_info(path):
    with open(path, "r") as f:
        ds_info = json.load(f)
    ds_info = {k.replace("/", "_"): v for k, v in ds_info.items()}
    ds_info = pd.DataFrame(ds_info).T
    return ds_info


def load_all_datasetnames_n_info(path, verbose=False):
    ds_list = parse_datasets(path)
    ds_list = list(map(lambda x: x.replace("/", "_"), ds_list))
    if verbose:
        print(ds_list, len(ds_list))

    ds_info = load_ds_info(ds_info_file)
    ds_info = ds_info.loc[ds_list, :]
    return ds_list, ds_info


def get_fmt_name(ds, ds_info):
    return ds_info.loc[ds]["name"] + " (" + ds_info.loc[ds]["domain"] + ")"


def pp_storing_path(path, save):
    if not isinstance(path, Path):
        path = Path(path)
    if save:
        path.mkdir(parents=True, exist_ok=True)
        print()
    return path


def print_interesting_columns(df):
    interesting_cols = []
    for col in df:
        if df[col].nunique() > 1:
            interesting_cols.append(col)
    display(df[interesting_cols])


def get_base_model(model_ids):
    return eval(model_ids)[0].split("@", 1)[0].rsplit("_", 1)[0]


def get_layer_types(model_ids):
    if "_ap@" in model_ids and "_cls@" in model_ids:
        return "cls+avg_pool"
    elif "_ap@" in model_ids:
        return "avg_pool"
    elif "_at@" in model_ids:
        return "all_tokens_last_layer"
    else:
        return "cls"


def get_model_size(base_model):
    mid = base_model.lower().replace("-", "_")
    mid = mid.replace("_b_16_", "_base_").replace("_l_", "_large_")
    if "small" in mid or "b_32" in mid:
        return "small"
    elif "base" in mid:
        return "base"
    elif "large" in mid:
        return "large"
    else:
        raise ValueError("No valid model size in base_model name={base_model}")


EXPERIMENT_CONFIG = {
    1: {
        "all_tokens_last_layer": "All tokens last layer",
        "avg_pool": "AP last layer",
        "cls": "CLS last layer",
    },
    2: {
        "cls+avg_pool": "CLS+AP last layer",
        "avg_pool": "AP layers from middle & last blocks",
        "cls": "CLS layers from middle & last blocks",
    },
    4: {
        "cls+avg_pool": "CLS+AP layers from middle & last blocks",
        "avg_pool": "AP layers from quarterly blocks",
        "cls": "CLS layers from quarterly blocks",
        "all_tokens_last_layer": "All tokens from quarterly blocks",
    },
    8: {
        "cls+avg_pool": "CLS+AP layers from quarterly blocks",
    },
    12: {"avg_pool": "AP layers from all blocks", "cls": "CLS layers from all blocks"},
    24: {
        "cls+avg_pool": "CLS+AP layers from all blocks",
        "avg_pool": "AP layers from all blocks",
        "cls": "CLS layers from all blocks",
    },
    48: {"cls+avg_pool": "CLS+AP layers from all blocks"},
}


def get_experiment_name(df):
    for row_idx, row in df.iterrows():
        task = row["task"]
        nr_layers = row["nr_layers"]
        curr_tokens = row["layer_types"]
        if "pipeline_type" in row and "finetuning" in row["pipeline_type"]:
            suffix = "finetuning"
        else:
            suffix = task.split("_")[0]

        try:
            exp_name = EXPERIMENT_CONFIG[nr_layers][curr_tokens]
        except KeyError as e:
            raise ValueError(
                f"No experiment name for nr_layers={nr_layers}, task={task} and token={curr_tokens}"
            ) from e

        df.loc[row_idx, "experiment"] = exp_name

        mode = row["mode"]

        if mode in ["combined_models", "end_2_end"]:
            df.loc[row_idx, "Experiment"] = f"{exp_name} ({suffix})"
        else:
            df.loc[row_idx, "Experiment"] = exp_name

    return df


DS_INFO = load_ds_info(ds_info_file)


def map_ds_name(x):
    if x in DS_INFO.index:
        return DS_INFO.loc[x, "name"]
    x_fmt = x.replace("/", "_")
    if x_fmt in DS_INFO.index:
        return DS_INFO.loc[x_fmt, "name"]
    else:
        raise ValueError(f"Dataset {x} not in the loaded dataset info")


def map_ds_domain(x):
    if x in DS_INFO.index:
        return DS_INFO.loc[x, "domain"]
    x_fmt = x.replace("/", "_")
    if x_fmt in DS_INFO.index:
        return DS_INFO.loc[x_fmt, "domain"]
    else:
        raise ValueError(f"Dataset {x} not in the loaded dataset info")


def safe_call(func):
    try:
        return func()
    except Exception:
        return np.nan


def add_additional_info(df):
    df["base_model"] = safe_call(lambda: df["model_ids"].apply(get_base_model))
    df["base_model_fmt"] = safe_call(
        lambda: df["base_model"].map(base_model_name_mapping)
    )
    df["dataset_fmt"] = safe_call(lambda: df["dataset"].apply(map_ds_name))
    df["dataset_domain"] = safe_call(lambda: df["dataset"].apply(map_ds_domain))
    df["layer_types"] = safe_call(lambda: df["model_ids"].apply(get_layer_types))
    df["nr_layers"] = safe_call(lambda: df["model_ids"].apply(lambda x: len(eval(x))))
    df["model_size"] = safe_call(lambda: df["base_model"].apply(get_model_size))
    df["hopt_time_hr"] = safe_call(lambda: pd.to_timedelta(df["hopt_time_s"], unit="s"))
    df["training_time_hr"] = safe_call(
        lambda: pd.to_timedelta(df["training_time"], unit="s")
    )
    df["train_data_inference_time_hr"] = safe_call(
        lambda: pd.to_timedelta(df["train_data_inference_time"], unit="s")
    )
    df["test_data_inference__time_hr"] = safe_call(
        lambda: pd.to_timedelta(df["test_data_inference_time"], unit="s")
    )
    try:
        df = get_experiment_name(df)
    except Exception as e:
        print("Error in get_experiment_name:", e)

    return df


def filter_df_for_best_runs(
    df: pd.DataFrame,
    metric_col: str = "best_val_bal_acc1",
    group_cols: list[str] = ["task", "experiment", "dataset", "model_ids"],
):
    runs = []
    for group_name, group_data in df.groupby(group_cols):
        if len(group_data) > 1:
            if any(group_data[metric_col].isna()):
                print(group_data[group_cols + [metric_col]])
            to_add = group_data.loc[group_data[metric_col].idxmax(), :]
        else:
            to_add = group_data.iloc[0, :]
        runs.append(to_add)
    print("df.shape before filtering for best runs", df.shape)
    df = pd.concat(runs, axis=1).T.reset_index(drop=True)
    print("df.shape after filtering for best runs", df.shape)
    return df


def get_abs_rel_performance(
    x: pd.DataFrame,
    ref_experiment: str = "CLS last layer",
    metric_columns: list[str] = [
        "train_lp_acc1",
        "train_lp_acc5",
        "train_lp_bal_acc1",
        "train_lp_bal_acc5",
        "test_lp_acc1",
        "test_lp_acc5",
        "test_lp_bal_acc1",
        "test_lp_bal_acc5",
        "best_val_bal_acc1",
    ],
):
    base_model = x.name[1]
    if "mae-vit" in base_model:
        ref_experiment = "AP last layer"
    ref_row = x[
        (x["experiment"] == ref_experiment)
        & (x["task"] == "linear_probe")
        & (x["mode"] == "single_model")
    ]
    ref_row = ref_row[
        ~ref_row["model_ids"].apply(
            lambda x: len(
                [
                    elem
                    for elem in eval(x)
                    if elem.split("@")[-1] not in ["norm", "visual"]
                ]
            )
            > 0
        )
    ]
    assert (
        len(ref_row) == 1
    ), f"Expected 1 row for ref_experiment={ref_experiment}, but got {ref_row=}"
    ref_row = ref_row.iloc[0]
    for metric in metric_columns:
        x[f"abs_perf_gain_{metric}"] = x[metric] - ref_row[metric]
    return x


def get_relative_performances_one_row(one_row, ref_df):
    metric_columns = [
        "train_lp_acc1",
        "train_lp_acc5",
        "train_lp_bal_acc1",
        "train_lp_bal_acc5",
        "test_lp_acc1",
        "test_lp_acc5",
        "test_lp_bal_acc1",
        "test_lp_bal_acc5",
    ]

    ds = one_row["dataset"]
    base_model = one_row["base_model"]
    idx = (ds, base_model)

    try:
        baseline = ref_df.loc[idx]
    except KeyError as e:
        print(e)
        return {f"relative_{col}": None for col in metric_columns}

    res = {}
    for col in metric_columns:
        curr_val = one_row[col]
        ref_val = baseline[col]
        res[f"relative_{col}"] = curr_val / ref_val
    return res


def get_relative_performances(df, ref_df):
    # preparation things
    if ref_df.index.names != ["dataset", "base_model"]:
        ref_df = ref_df.set_index(["dataset", "base_model"])

    df_rel_perfs = df.apply(get_relative_performances_one_row, ref_df=ref_df, axis=1)
    df_rel_perfs = pd.DataFrame(df_rel_perfs.tolist())
    df_with_rel_perfs = pd.concat([df, df_rel_perfs], axis=1)
    return df_with_rel_perfs


def set_ylims_with_margin(data, **kwargs):
    """Set y-axis limits with percentage margin"""
    ax = plt.gca()
    curr_data = data[kwargs["metric"]]
    data_min = np.min(curr_data)
    data_max = np.max(curr_data)
    data_range = data_max - data_min

    margin = data_range * kwargs["margin_percent"]

    if "relative" in kwargs["metric"]:
        ymin = min(data_min - margin, 0.95)
        ymax = max(data_max + margin, 1.05)
        ax.axhline(1, c="grey", ls=":", zorder=-1)
    else:
        ymin = data_min - margin
        ymax = data_max + margin

    ax.set_ylim((ymin, ymax))


def style_multimodel_heatmap(
    df,
    color_maps=["Reds", "Blues", "Greens", "Purples", "Oranges", "YlOrBr"],
    precision=4,
):
    models = df.columns.get_level_values(0).unique()

    styler = df.style

    for i, model in enumerate(models):
        model_cols = df.columns[df.columns.get_level_values(0) == model]
        cmap = color_maps[i % len(color_maps)]

        styler = styler.background_gradient(
            cmap=cmap,
            subset=model_cols,
            axis=1,  # Compare within each column
        )
    styler = styler.format(precision=precision, na_rep="")
    styler = styler.apply(
        lambda x: ["background-color: white" if pd.isna(v) else "" for v in x], axis=1
    )
    return styler


def init_plotting_params():
    plt.rcParams["axes.titlesize"] = fontsizes["title"]
    plt.rcParams["legend.fontsize"] = fontsizes["legend"] - 1
    plt.rcParams["legend.title_fontsize"] = fontsizes["legend"]
    plt.rcParams["axes.labelsize"] = fontsizes["label"]
    plt.rcParams["xtick.labelsize"] = fontsizes["ticks"]
    plt.rcParams["ytick.labelsize"] = fontsizes["ticks"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    print(
        json.dumps(
            {k: v for k, v in plt.rcParams.items() if "font" in k or "size" in k},
            indent=2,
        )
    )
