import argparse
import os
from typing import Union, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple
import torch
from src.utils.model_mapping import get_hash_from_model_id
from sklearn.cluster import SpectralClustering
from src.data.data_utils import prepare_ds_name
import re
from loguru import logger


def check_paths(args: argparse.Namespace, dataset: str) -> Tuple[Path, Path]:
    """Check if the similarity matrix and model ids exist."""
    root = Path(args.sim_mat_root) / dataset / args.method_key
    subdir = get_hash_from_model_id(args.model_keys)
    base_dir = root / subdir
    sim_mat_path = base_dir / 'similarity_matrix.pt'
    model_ids_path = base_dir / 'model_ids.txt'

    if not sim_mat_path.exists():
        raise FileNotFoundError(f"Similarity matrix not found at {sim_mat_path}")
    if not model_ids_path.exists():
        raise FileNotFoundError(f"Model ids not found at {model_ids_path}")
    return sim_mat_path, model_ids_path


def load_similarity_matrix(sim_mat_path: Path, model_ids_path: Path):
    """Load the similarity matrix and model ids."""
    sim_mat = torch.load(sim_mat_path)
    with open(model_ids_path, 'r') as f:
        model_ids = f.read().splitlines()
    return pd.DataFrame(sim_mat, index=model_ids, columns=model_ids)


def process_similarity_matrix(sim_mat: pd.DataFrame, allowed_models: Optional[List[str]]=None) -> pd.DataFrame:
    """Process the similarity matrix:
    1. Filter only desired models
    2. Take the absolute value of the matrix
    3. Fill diagonal with 1
    4. Fill NaNs with 0
    """
    # filter only desired models
    if allowed_models:
        sim_mat = sim_mat.loc[allowed_models, allowed_models].copy()

    sim_mat = sim_mat.abs()
    if not np.all(np.diag(sim_mat.values) == 1):
        np.fill_diagonal(sim_mat.values, 1)

    if sim_mat.isnull().values.any():
        logger.warning("Similarity matrix contains NaNs, filling them with 0.")
        sim_mat = sim_mat.fillna(0)
    return sim_mat


def remap_clusters_by_first_occurrence(series: pd.Series) -> pd.Series:
    mapping = {}

    for val in series:
        if val not in mapping:
            mapping[val] = len(mapping)  # Assign next available ID

    return series.map(mapping)


def sort_cluster_key(model_id):
    # Split by '@'
    parts = model_id.split('@')
    suffix = parts[1] if len(parts) > 1 else ''

    # Find the first number surrounded by dots, e.g. ".10."
    match = re.search(r'\.(\d+)\.', suffix)
    block_num = int(match.group(1)) if match else np.inf

    # _ap before _cls, else last
    if '_ap' in parts[0]:
        type_order = 0
    elif '_cls' in parts[0]:
        type_order = 1
    else:
        type_order = 2

    # Extract trailing number from norm layers for tie-breaking
    norm_match = re.search(r'(?:norm|ln_)?(\d+)$', suffix)
    norm_order = int(norm_match.group(1)) if norm_match else -1  # -1 if none, so they come first

    return (block_num, norm_order, type_order)

def get_model_combinations_from_clustering(
        clustering_root: str,
        dataset: str,
        method_key: str,
        num_clusters: int,
        model_ids: List[str],
        assign_labels: str ="kmeans"
    ) -> List[str]:
    subdir = get_hash_from_model_id(model_ids)
    output_path = Path(clustering_root) / dataset / method_key / subdir / f"num_clusters_{num_clusters}" / assign_labels
    labels_path = output_path / 'cluster_labels.csv'
    if not os.path.exists(labels_path):
        # TODO Mayne we just want to create the clusters if they do not exist? Would require seed if not hardcoded!
        raise FileNotFoundError(f"Cluster labels file {labels_path} does not exist.")

    labels_df = pd.read_csv(labels_path)
    # Always take the last module from each cluster
    model_combinations = []
    for cluster_id, cluster_data in labels_df.groupby('cluster', sort=False):
        model_combinations.append(cluster_data['model_id'].iloc[-1])
    return model_combinations

def main(args: argparse.Namespace):
    for dataset in args.datasets:
        dataset = prepare_ds_name(dataset)
        sim_mat_path, model_ids_path = check_paths(args, dataset)
        sim_mat = load_similarity_matrix(sim_mat_path, model_ids_path)
        # TODO currently disable the allowed model selection and cluster across the whole similarity matrix
        sim_mat = process_similarity_matrix(sim_mat, allowed_models=None)
        subdir = get_hash_from_model_id(args.model_keys)
        for n_clusters in args.num_clusters:
            # Check if the clustering results already exist
            output_path = Path(args.clustering_root) / dataset / args.method_key/ subdir /f"num_clusters_{n_clusters}"/args.assign_labels
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / 'cluster_labels.csv'

            if output_file.exists() and not args.overwrite:
                logger.info(f"Skipping {n_clusters} clusters; it already exists at {output_file=} and --overwrite is not set.")
                continue

            if n_clusters > len(sim_mat):
                logger.warning(f"Skipping {n_clusters} clusters; it exceeds the number of models ({len(sim_mat)}) ")
                continue

            clustering = SpectralClustering(n_clusters=n_clusters,
                                            affinity='precomputed',
                                            assign_labels=args.assign_labels,
                                            random_state=args.seed)
            labels = clustering.fit_predict(sim_mat.values, y=None)
            labels = pd.DataFrame({'model_id': sim_mat.index.to_numpy(), 'cluster': labels}, index=sim_mat.index)
            
            # Sorting the cluster by block structure
            labels = labels.assign(sort_key=labels['model_id'].map(sort_cluster_key)) \
                .sort_values('sort_key') \
                .drop(columns='sort_key')

            labels["cluster"] = remap_clusters_by_first_occurrence(labels["cluster"])

            labels.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', nargs="+",default=[-1], type=int,
                        help="List of number of clusters to create.")
    parser.add_argument('--method_key', type=str,
                        choices=[
                            'cka_kernel_rbf_unbiased_sigma_0.2',
                            'cka_kernel_rbf_unbiased_sigma_0.4',
                            'cka_kernel_rbf_unbiased_sigma_0.6',
                            'cka_kernel_rbf_unbiased_sigma_0.8',
                            'cka_kernel_linear_unbiased',
                            'rsa_method_correlation_corr_method_pearson',
                            'rsa_method_correlation_corr_method_spearman',
                            'cosine_similarity',
                        ],
                        default='rsa_correlation_spearman')

    parser.add_argument('--datasets', type=str, default=['imagenet-subset-10k',], nargs='+',
                        help="Datasets to use for clustering.")
    parser.add_argument('--model_keys', type=str, nargs='+', help="Define the models we want to consider during clustering.")
    parser.add_argument('--assign_labels', type=str, default='kmeans',
                        choices=['kmeans', 'discretize', 'cluster_qr'],
                        help="Method used to assign labels during SpectralClustering.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument('--sim_mat_root', type=str)
    parser.add_argument('--clustering_root', type=str)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)
