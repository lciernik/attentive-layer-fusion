from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from functools import partial
from loguru import logger


class BaseFeatureCombiner:
    """Base class for feature combiners."""

    def __init__(self, reference_combiner: Union["BaseFeatureCombiner", None] = None) -> None:
        """Initialize the feature combiner.

        Args:
            reference_combiner: A reference combiner to use for initialization.
        """
        self.features = None
        self.reference_combiner = reference_combiner

    def __getitem__(self, i: int) -> torch.Tensor:
        """Get the feature at index i."""
        return self.features[i]

    def __call__(self, i: int) -> torch.Tensor:
        """Call the feature combiner at index i."""
        return self.features[i]

    def get_feature_dim(self) -> int:
        """Get the dimension of the features."""
        return self.features.shape[1]

    def set_features(self, list_features: list[torch.Tensor]) -> None:
        """Set the features."""
        self.features = list_features

class ConcatFeatureCombiner(BaseFeatureCombiner):
    """Concatenate the features of the list of features."""

    def set_features(self, list_features: list[torch.Tensor]) -> None:
        """Set the features."""
        self.features = torch.concat(list_features, dim=1)


class StackedZeroPadFeatureCombiner(BaseFeatureCombiner):
    """Stack the features of the list of features."""
    def __init__(self,  reference_combiner: Union["BaseFeatureCombiner", None] = None, shared_dim: int | None = None) -> None:
        super().__init__(reference_combiner=reference_combiner)
        self.shared_dim = shared_dim
        self.already_stacked = None

    def set_features(self, list_features: list[torch.Tensor]) -> None:
        """Creates a huge tensor with all the model representations.
        By default, it pads the features with zeros to the shared dimension.
        If the shared dimension is not provided, it will be detemined by the maximum dimension of the input features.
        """
            
        max_dim = max([feat.shape[1] for feat in list_features]) # Assume list_features=[(N, d1), (N, d2), ... (N, dk)]
            
        if self.shared_dim is None:
            self.shared_dim = max(feat.shape[1] for feat in list_features)
        elif self.shared_dim < max_dim:
            raise ValueError(f"{self.shared_dim=} must be larger or equal to the {max_dim=} of features.")

        list_features = [F.pad(feat, (0, self.shared_dim - feat.shape[1], 0, 0)) for feat in list_features]
        print("Setting features in StackedZeroPadFeatureCombiner with shape:", list_features[0].shape)
        # if list_features[0].shape[0] > 1_000_000:
        if list_features[0].shape[0] > 250_000:
            # We have imagenetand can't stack all at once
            logger.info("Stacking features in chunks to avoid memory issues.")
            self.features = list_features
            self.already_stacked = False
        else:
            self.features = torch.stack(list_features, dim=1) # Get (N, len(list_features) , d_shared)
            self.already_stacked = True

    def get_feature_dim(self) -> list[int]:
        """Get the dimension of the features."""
        if not self.already_stacked:
            return [self.shared_dim] * len(self.features)
        return [self.features.shape[2]] * self.features.shape[1]

    def __getitem__(self, i: int) -> torch.Tensor:
        """Get the feature at index i."""
        if self.already_stacked:
            return self.features[i]
        else:
            return torch.stack([feat[i] for feat in self.features], dim=0)

    def __call__(self, i: int) -> torch.Tensor:
        """Call the feature combiner at index i."""
        if self.already_stacked:
            return self.features[i]
        else:
            return torch.stack([feat[i] for feat in self.features], dim=0)



class AlreadyStackedFeatureCombiner(BaseFeatureCombiner):
    def __init__(self,  reference_combiner: Union["BaseFeatureCombiner", None] = None, shared_dim: int | None = None) -> None:
        super().__init__(reference_combiner=reference_combiner)
        self.shared_dim = shared_dim

    def set_features(self, list_features: list[torch.Tensor]) -> None:
        """Set the features."""
        if not isinstance(list_features[0], torch.Tensor):
            raise ValueError("AlreadyStackedFeatureCombiner expects a single feature tensor.")
        if list_features[0].ndim != 3:
            raise ValueError("AlreadyStackedFeatureCombiner expects a 3D feature tensor, with shape (N, T, D), where N is the number of samples, T is the number of tokens, and D is the feature dimension.")
        if self.shared_dim < list_features[0].shape[2]:
            raise ValueError(f"{self.shared_dim=} must be larger or equal to the {list_features[0].shape[2]=} of features.")
        if len(list_features) != 1:
            logger.info("We have multiple already stacked feature, we assume they have all 3 dims!")
            list_features = [F.pad(feat, (0, self.shared_dim - feat.shape[2], 0, 0)) for feat in list_features]
            self.features = torch.cat(list_features,dim=1)
            #raise ValueError("AlreadyStackedFeatureCombiner expects a single feature tensor.")
        else:
            self.features = list_features[0]
            self.features = F.pad(self.features, (0, self.shared_dim - self.features.shape[2], 0, 0))
        logger.info(f"AlreadyStackedFeatureCombiner: {self.features.shape}")

    def get_feature_dim(self) -> list[int]:
        """Get the dimension of the features."""
        return [self.features.shape[2]] * self.features.shape[1]


class PCAConcatFeatureCombiner(BaseFeatureCombiner):
    """Concatenate the features of the list of features and apply PCA."""

    def __init__(
        self, pct_var: float = 0.99, reference_combiner: Union["PCAConcatFeatureCombiner", None] = None
    ) -> None:
        """Initialize the PCAConcatFeatureCombiner.

        Args:
            pct_var: The percentage of variance to keep.
            reference_combiner: A reference combiner to use for initialization. In particular, this is used to fit the
            PCA on train data and then use the same PCA for test data.
        """
        super().__init__()
        if reference_combiner is None:
            self.pca = PCA()
            self.scalar = StandardScaler()
            self.pct_var = pct_var
            self.n_components = None
            self.scale_fn = self.scalar.fit_transform
            self.pca_fn = self.pca.fit_transform

        else:
            if not isinstance(reference_combiner, PCAConcatFeatureCombiner):
                raise ValueError("Reference combiner should be a PCAConcatFeatureCombiner")

            if reference_combiner.features is None:
                raise ValueError("Reference combiner should have features set")

            self.pca = reference_combiner.pca
            self.scalar = reference_combiner.scalar
            self.pct_var = reference_combiner.pct_var
            self.n_components = reference_combiner.n_components
            self.scale_fn = self.scalar.transform
            self.pca_fn = self.pca.transform

    def set_features(self, list_features: list[torch.Tensor]) -> None:
        """Set the features."""
        features = torch.concat(list_features, dim=1)
        scaled_features = self.scale_fn(features)
        pca_features = self.pca_fn(scaled_features)
        if self.n_components is None:
            self.n_components = np.argmax(np.cumsum(self.pca.explained_variance_ratio_) > self.pct_var) + 1
        self.features = torch.Tensor(pca_features[:, : self.n_components])

class TupleFeatureCombiner(BaseFeatureCombiner):
    """Return a tuple of features for each index."""

    def set_features(self, list_features: list[torch.Tensor]) -> None:
        """Set the features."""
        self.features = list_features

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        """Get the feature at index i."""
        return tuple(self.features[j][i] for j in range(len(self.features)))

    def __call__(self, i: int) -> tuple[torch.Tensor, ...]:
        """Call the feature combiner at index i."""
        return tuple(self.features[j][i] for j in range(len(self.features)))

    def get_feature_dim(self) -> list[int]:
        """Get the dimension of the features."""
        return [self.features[j].shape[1] for j in range(len(self.features))]
    



def get_feature_combiner_cls(combiner_name: str, shared_dim: int | None = None) -> BaseFeatureCombiner:
    """Get the feature combiner class based on the combiner name."""
    if combiner_name == "concat":
        return ConcatFeatureCombiner
    elif combiner_name == "concat_pca":
        return PCAConcatFeatureCombiner
    elif combiner_name == "tuple":
        return TupleFeatureCombiner
    elif combiner_name == "stacked_zero_pad":
        return partial(StackedZeroPadFeatureCombiner, shared_dim=shared_dim)
    elif combiner_name == "already_stacked_zero_pad":
        return partial(AlreadyStackedFeatureCombiner, shared_dim=shared_dim)
    else:
        raise ValueError(f"Unknown feature combiner name: {combiner_name}")
