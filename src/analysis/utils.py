from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.utils import retrieve_model_dataset_results

HYPER_PARAM_COLS = ["model_ids", "fewshot_k", "epochs", "batch_size"]


def retrieve_performance(
    model_id: str,
    dataset_id: str,
    metric_column: str = "test_lp_acc1",
    results_root: str = "/home/space/diverse_priors/results/linear_probe/single_model",
    regularization: str = "weight_decay",
    allow_db_results: bool = True,
) -> float | np.floating:
    """Retrieve the performance of a model on a dataset."""
    path = Path(results_root) / dataset_id / model_id

    df = retrieve_model_dataset_results(path, allow_db_results=allow_db_results)

    if df.dataset.nunique() > 1:
        raise ValueError(f"Result files for {model_id=} and {dataset_id=} contain multiple datasets. Cannot proceed.")

    # filter regularization method
    if "regularization" not in df.columns:
        raise ValueError("Regularization was not available yet.")

    df = df[df.regularization == regularization]

    if len(df) == 0:
        raise ValueError(f"No results available for {dataset_id=}, {model_id=} and {regularization=}.")

    # TODO: Remove manually filtering for seeds # noqa: TD002, TD003
    if df["seed"].nunique() > 3:
        logger.warning(
            f"More than 3 seeds ({df['seed'].unique()}) available for {model_id=} and {dataset_id=}. "
            f"We will filter for the first 3 seeds."
        )
    df = df[df["seed"].isin([0, 1, 2])]

    performance = df.groupby(HYPER_PARAM_COLS)[metric_column].mean()

    # TODO: Remove allowing for more than one configuration setting # noqa: TD002, TD003
    if len(performance) > 1:
        logger.error(performance)
        raise ValueError(
            "We have more than one configuration setting for the model and dataset. Since we are only "
            "considering one configuration setting per regularization method, we do not proceed."
        )

    return performance.max()
