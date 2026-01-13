from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.data_utils import GaussianNoise


class BaseActivationCombiner(nn.Module, ABC):
    """Base class to combine activations from a premodel."""

    def __init__(self, shared_dim: int | None = None, jitter_p: float = 0.5, normalize: bool = True):
        super().__init__()
        self.shared_dim = shared_dim
        self.normalize = normalize
        self.transform = GaussianNoise(mean=0.0, std=0.05, p=jitter_p)

    @abstractmethod
    def forward(self, acts: list[torch.Tensor]) -> torch.Tensor:
        """Postprocess the activations collected by the hook based on the method"""
        raise NotImplementedError


class ConcatActivationCombiner(BaseActivationCombiner):
    """Concatenate the activations of the list.
    NOTE: No padding is done in this setting."""

    def forward(self, acts: list[torch.Tensor]) -> torch.Tensor:
        """Postprocess the activations collected by the hook based on the method"""
        if self.normalize:
            acts = [F.normalize(act, p=2, dim=-1) for act in acts]
        concat_acts = torch.cat(acts, dim=1)
        if len(concat_acts.shape) > 2:
            concat_acts = concat_acts.transpose(1, 2).flatten(1)  # |[(B, tokens, dim)]| = L -> (B, L*tokens*dim)
        if self.training:
            concat_acts = self.transform(concat_acts)
        return concat_acts


class StackZeroPadActivationCombiner(BaseActivationCombiner):
    """Concatenate the activations of the list."""

    def forward(self, acts: list[torch.Tensor]) -> torch.Tensor:
        """Postprocess the activations collected by the hook based on the method"""
        if self.normalize:
            acts = [F.normalize(act, p=2, dim=-1) for act in acts]

        if self.shared_dim is None:
            raise ValueError("Cannot process with activation combination as self.shared_dim=None.")
        padded_acts = [F.pad(act, (0, self.shared_dim - act.shape[-1], 0, 0)) for act in acts]
        if len(padded_acts[0].shape) == 3:
            stacked_acts = torch.cat(padded_acts, dim=1)
        else:
            stacked_acts = torch.stack(padded_acts, dim=1)

        if self.training:
            stacked_acts = self.transform(stacked_acts)

        return stacked_acts


def get_activation_combiner(
    combiner_name: str,
    shared_dim: int | None = None,
    jitter_p: float = 0.0,
    normalize: bool = False,
) -> BaseActivationCombiner:
    """Get the feature combiner class based on the combiner name."""
    if combiner_name == "concat":
        return ConcatActivationCombiner(jitter_p=jitter_p, normalize=normalize)
    elif combiner_name == "stacked_zero_pad":
        return StackZeroPadActivationCombiner(shared_dim=shared_dim, jitter_p=jitter_p, normalize=normalize)
    else:
        raise ValueError(f"Unknown feature combiner name: {combiner_name}")
