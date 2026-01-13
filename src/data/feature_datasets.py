import torch
from torch.utils.data import Dataset

__all__ = ["CombinedFeaturesDataset", "FeatureDataset", "Rep2RepFeaturesDataset"]


class FeatureDataset(Dataset):
    """Dataset for a single model's representations."""

    def __init__(self, features, targets, transform=None):
        self.features = features
        self.targets = targets
        self.transform = transform

    @property
    def feature_dims(self) -> int:
        return self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        features = self.features[i]
        if self.transform is not None:
            features = self.transform(features)
        return features, self.targets[i]


class CombinedFeaturesDataset(Dataset):
    """Dataset for multiple models' representations."""

    def __init__(self, list_features, targets, feature_combiner, transform=None):
        if not isinstance(list_features, list):
            list_features = [list_features]
        self.targets = targets
        self.nr_comb_feats = len(list_features)
        self.nr_samples = len(list_features[0])
        self.feature_combiner = feature_combiner
        self.feature_combiner.set_features(list_features)
        self.transform = transform

    @property
    def feature_dims(self) -> list[int]:
        return self.feature_combiner.get_feature_dim()

    def __len__(self):
        return self.nr_samples

    def __getitem__(self, i):
        features = self.feature_combiner(i)
        if self.transform is not None:
            features = self.transform(features)
        return features, self.targets[i]


class Rep2RepFeaturesDataset(Dataset):
    """Dataset for two models' representations."""

    def __init__(self, list_features, targets):
        assert len(list_features) == 2, "Rep2RepFeaturesDataset requires exactly 2 Model features."
        self.targets = targets
        self.features, self.target_reps = list_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.target_reps[i], self.targets[i]
