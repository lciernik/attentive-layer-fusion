import random
import warnings
from collections.abc import Callable
from itertools import chain

import numpy as np
from loguru import logger


class Sampler:
    """Base class for all samplers."""

    def __init__(self, k: int, models: dict, seed: int = 0) -> None:
        self.k = k
        self.models = models
        random.seed(seed)

    def sample(self) -> list[str]:
        """Sample a set of models from the sampler."""
        raise NotImplementedError("not implemented")

    def max_available_samples(self) -> int | float:
        """Get the maximum number of samples that can be drawn from the sampler."""
        return np.inf


class TopKSampler(Sampler):
    """Sampler that samples the top k models based on a scoring function."""

    def __init__(self, model_scoring_fn: Callable, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_scoring_fn = model_scoring_fn

    def sample(self) -> list[str]:
        """Sample a set of models from the sampler. We want to have models with
        the highest score/performance first. Take the top k.
        """
        model_ids = list(self.models.keys())

        ranked_models = sorted(model_ids, key=lambda model_id: self.model_scoring_fn(model_id), reverse=True)
        # take first k
        return ranked_models[: self.k]

    def max_available_samples(self) -> int | float:
        """Get the maximum number of samples that can be drawn from the sampler.
        This sampler always returns the same model set.
        """
        return 1


class RandomSampler(Sampler):
    """Samples totally random from the available models"""

    def sample(self) -> list[str]:
        """Sample a set of models from the sampler. Take random k models."""
        selected_models = random.sample(list(self.models.keys()), k=self.k)
        return selected_models


class BaseClusterSampler(Sampler):
    """Base class for all cluster samplers."""

    def __init__(
        self,
        cluster_assignment: dict[int, list[str]],
        model_scoring_fn: Callable | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cluster_assignment = cluster_assignment
        self.model_scoring_fn = model_scoring_fn
        self._check_model_ids_n_clusters_assignment()

    def _check_model_ids_n_clusters_assignment(self) -> None:
        model_ids = list(self.models.keys())
        set_model_ids = set(model_ids)
        set_model_dict_ids = set(chain.from_iterable(self.cluster_assignment.values()))
        ids_intersection = set_model_ids.intersection(set_model_dict_ids)

        new_model_ids = sorted(ids_intersection)
        if len(model_ids) - len(new_model_ids) > 0:
            logger.warning(
                f"Removing models ({set_model_ids.difference(ids_intersection)}) "
                f"not present in the clustering assignments"
            )
        self.models = {k: v for k, v in self.models.items() if k in new_model_ids}

        if len(set_model_dict_ids) - len(ids_intersection) > 0:
            logger.warning(
                f"Removing models ids from cluster assignment ({set_model_dict_ids.difference(ids_intersection)}) "
                "that are not in the available models."
            )
        self.cluster_assignment = {
            k: sorted([m for m in val if m in ids_intersection]) for k, val in self.cluster_assignment.items()
        }
        self.cluster_assignment = {k: v for k, v in self.cluster_assignment.items() if v}

    def _get_mean_cluster_score(self, cluster_id: int) -> float | np.floating:
        if self.model_scoring_fn is None:
            raise ValueError("model_scoring_fn needs to be specified")
        cluster_models = self.cluster_assignment[cluster_id]
        cluster_scores = [self.model_scoring_fn(model_id) for model_id in cluster_models]
        return np.mean(cluster_scores)

    def rank_clusters_by_mean_score(self) -> list[int]:
        """Rank clusters by their mean score."""
        cluster_scores = {
            cluster_id: self._get_mean_cluster_score(cluster_id) for cluster_id in self.cluster_assignment
        }
        ranked_cluster_ids = sorted(cluster_scores, key=lambda cluster_id: cluster_scores[cluster_id], reverse=True)
        return ranked_cluster_ids

    def sample(self) -> list[str]:
        """Sample a set of models from the sampler."""
        raise NotImplementedError("Method `sample` not implemented for BaseClusterSampler. Use subclasses instead.")


class ClusterSampler(BaseClusterSampler):
    """Sampler that samples a set of clusters and then samples k models from each cluster."""

    def __init__(self, selection_strategy: str = "random", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.selection_strategy = selection_strategy
        if self.selection_strategy not in ["random", "best"]:
            raise ValueError("selection strategy should be either random or best")

        if self.selection_strategy == "best" and self.model_scoring_fn is None:
            raise ValueError("model_scoring_fn needs to be specified")

    def get_selected_clusters(self) -> list[int]:
        """Get the selected clusters."""
        if len(self.cluster_assignment) < self.k:
            raise ValueError("ClusterSampler requires at least k clusters in self.cluster_assignment to work.")
        ranked_cluster_ids = self.rank_clusters_by_mean_score()
        selected_clusters = ranked_cluster_ids[: self.k]
        return selected_clusters

    def sample(self) -> list[str]:
        """Sample a set of models from the sampler."""
        selected_clusters = self.get_selected_clusters()
        model_set = []
        for cluster in selected_clusters:
            if self.selection_strategy == "random":
                selected_model = random.choice(self.cluster_assignment[cluster])
            elif self.selection_strategy == "best":
                ranked_models = sorted(
                    self.cluster_assignment[cluster],
                    key=lambda model_id: self.model_scoring_fn(model_id),
                    reverse=True,
                )
                selected_model = ranked_models[0]
            else:
                raise ValueError("unknown model selection strategy")
            model_set.append(selected_model)
        return model_set

    def max_available_samples(self) -> int | float:
        """Get the maximum number of samples that can be drawn from the sampler."""
        return np.inf if self.selection_strategy == "random" else 1


class OneClusterSampler(BaseClusterSampler):
    """Sampler that samples one sample from a single cluster."""

    def __init__(self, cluster_index: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cluster_index = cluster_index

    def get_selected_cluster(self) -> int:
        """Get the selected cluster."""
        if len(self.cluster_assignment) == 0:
            raise ValueError("No cluster found.")

        ranked_cluster_ids = self.rank_clusters_by_mean_score()
        # Filter out clusters with less than k models
        selected_clusters = [
            cluster for cluster in ranked_cluster_ids if len(self.cluster_assignment[cluster]) >= self.k
        ]
        if len(selected_clusters) == 0:
            raise ValueError("no cluster with at least k models found")
        return selected_clusters[0]

    def sample(self) -> list[str]:
        """Sample a set of models from the sampler."""
        model_options = self.cluster_assignment[self.cluster_index]
        if self.k > len(model_options):
            warnings.warn(
                "Number of selected models is larger than cluster size, limiting to cluster size..", stacklevel=2
            )
        model_set = random.sample(model_options, k=min(self.k, len(model_options)))
        return model_set
