import re
import warnings
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from thingsvision.core.cka.cka_base import CKABase

Array = np.ndarray


class Regularization(Enum):
    L1 = "L1"
    L2 = "L2"
    weight_decay = "weight_decay"

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]


class Regularizer:
    def __init__(self, reg_type: str, reg_lambda: float):
        try:
            self.reg_type = Regularization(reg_type)
        except ValueError as e:
            raise ValueError(
                f"Regularization type {reg_type} not supported. Choose from {list(Regularization)}"
            ) from e
        self.reg_lambda = reg_lambda
        self.reg_func = self._get_regularizer(self.reg_type)

    @staticmethod
    def _get_regularizer(reg_type: Regularization):
        if reg_type == Regularization.L1:
            return lambda x: torch.sum(torch.abs(x))
        elif reg_type == Regularization.L2:
            return lambda x: torch.sum(x**2)
        elif reg_type == Regularization.weight_decay:
            return lambda x: 0.0

    def reg_loss(self, model):
        if self.reg_type == Regularization.weight_decay:
            return 0
        reg_loss = sum([self.reg_func(param) for name, param in model.named_parameters() if "bias" not in name])
        return self.reg_lambda * reg_loss

    def get_lambda(self):
        return self.reg_lambda if self.reg_type == Regularization.weight_decay else 0.0


class CosineDistanceLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super(CosineDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distance = 1 - torch.nn.functional.cosine_similarity(pred, target)
        if self.reduction == "mean":
            return distance.mean()
        elif self.reduction == "sum":
            return distance.sum()
        else:
            return distance


class CKALoss(torch.nn.Module):
    def __init__(self, batch_size: int, device: str, type: str = "linear") -> None:
        super(CKALoss, self).__init__()

        if "rbf" in type:
            # Extract sigma from type
            sigma = float(type.split("_")[-1])
            type = "rbf"
        else:
            sigma = 1.0
        self.cka_g = CKATorch(m=batch_size, kernel=type, sigma=sigma, unbiased=True, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - self.cka_g.compare(X=pred, Y=target)


class CKATorch(CKABase):
    def __init__(
        self,
        m: int,
        kernel: str,
        unbiased: bool = False,
        device: str = "cpu",
        verbose: bool = False,
        sigma: Optional[float] = 1.0,
        compile: bool = True,
    ) -> None:
        """
        PyTorch implementation of CKA. Runs on CPU and CUDA.
        Args:
            m (int) - number of images / examples in a mini-batch or the full dataset;
            kernel (str) - 'linear' or 'rbf' kernel for computing the gram matrix;
            unbiased (bool) - whether to compute an unbiased version of CKA;
            device (str) - whether to perform CKA computation on CPU or GPU;
            sigma (float) - for 'rbf' kernel sigma defines the width of the Gaussian;
        """
        super().__init__(m=m, kernel=kernel, unbiased=unbiased, sigma=sigma)
        device = self._check_device(device, verbose)
        if device == "cpu" or not compile:
            self.hsic = self._hsic
        else:
            # use JIT compilation on CUDA
            self.hsic = torch.compile(self._hsic)
        self.device = torch.device(device)

    @staticmethod
    def _check_device(device: str, verbose: bool) -> str:
        """Check whether the selected device is available on current compute node."""
        if device.startswith("cuda"):
            gpu_index = re.search(r"cuda:(\d+)", device)

            if not torch.cuda.is_available():
                warnings.warn(
                    "\nCUDA is not available on your system. Switching to device='cpu'.\n",
                    category=UserWarning,
                )
                device = "cpu"
            elif gpu_index and int(gpu_index.group(1)) >= torch.cuda.device_count():
                warnings.warn(
                    f"\nGPU index {gpu_index.group(1)} is out of range. "
                    f"Available GPUs: {torch.cuda.device_count()}. "
                    f"Switching to device='cuda:0'.\n",
                    category=UserWarning,
                )
                device = "cuda:0"
        return device

    def centering(self, K: torch.Tensor) -> torch.Tensor:
        """Centering of the (square) gram matrix K."""
        # The below code block is mainly copied from Simon Kornblith's implementation;
        # see here: https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
        if torch.isnan(K).any() or torch.isinf(K).any():
            raise ValueError("Input matrix X contains NaN or Inf values.")

        if not torch.allclose(K, K.T, rtol=1e-02, atol=1e-03):
            raise ValueError("\nInput array must be a symmetric matrix.\n")
        if self.unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = K.shape[0]
            # K.fill_diagonal_(0.0)
            K = K - torch.diag(K.diag())  # Avoid in-place fill_diagonal_
            means = K.sum(dim=0) / (n - 2)
            means = means - means.sum() / (2 * (n - 1))
            K = K - means[:, None] - means[None, :]
            # K.fill_diagonal_(0.0)
            K = K - torch.diag(K.diag())  # Avoid in-place fill_diagonal_
        else:
            means = K.mean(dim=0)
            means = means - means.mean() / 2
            K = K - means[:, None] - means[None, :]
        return K

    def apply_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the (square) gram matrix K."""
        try:
            K = getattr(self, f"{self.kernel}_kernel")(X)
        except AttributeError:
            raise NotImplementedError
        return K

    def linear_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """Use a linear kernel for computing the gram matrix."""
        return X @ X.T

    def rbf_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """Use an rbf kernel for computing the gram matrix. Sigma defines the width."""
        GX = X @ X.T
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if self.sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = torch.sqrt(mdist)
        else:
            sigma = self.sigma
        KX = (KX * (-0.5 / sigma**2)).exp()
        return KX

    def _hsic(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        K = self.apply_kernel(X)
        L = self.apply_kernel(Y)
        K_c = self.centering(K)
        L_c = self.centering(L)
        """Compute the Hilbert-Schmidt independence criterion."""
        # np.sum(K_c * L_c) is equivalent to K_c.flatten() @ L_c.flatten() or in math
        # sum_{i=0}^{m} sum_{j=0}^{m} K^{\prime}_{ij} * L^{\prime}_{ij} = vec(K_c)^{T}vec(L_c)
        return torch.sum(K_c * L_c)

    @staticmethod
    def _check_types(
        X: Union[Array, torch.Tensor],
        Y: Union[Array, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(X, Array):
            X = torch.from_numpy(X)
        if isinstance(Y, Array):
            Y = torch.from_numpy(Y)
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError("Input X contains NaN or Inf values.")
        if torch.isnan(Y).any() or torch.isinf(Y).any():
            raise ValueError("Input Y contains NaN or Inf values.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same number of rows! Got {X.shape[0]} and {Y.shape[0]}.")
        return X, Y

    def compare(
        self,
        X: Union[Array, torch.Tensor],
        Y: Union[Array, torch.Tensor],
    ) -> torch.Tensor:
        """Compare two representation spaces X and Y using CKA."""
        X, Y = self._check_types(X, Y)
        # move X and Y to current device
        X = X.to(self.device)
        Y = Y.to(self.device)
        hsic_xy = self.hsic(X, Y)
        hsic_xx = self.hsic(X, X)
        hsic_yy = self.hsic(Y, Y)
        eps = 1e-4  # Small regularization constant
        rho = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy + eps))
        return rho


class CombinedLoss(torch.nn.Module):
    def __init__(self, loss_1, loss_2, alpha: float = 0.5) -> None:
        super(CombinedLoss, self).__init__()
        self.loss_1 = loss_1
        self.loss_2 = loss_2
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_1 = self.loss_1(pred, target)
        loss_2 = self.loss_2(pred, target)
        return self.alpha * loss_1 + (1 - self.alpha) * loss_2

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha

    def get_alpha(self) -> float:
        return self.alpha


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 1.0, temperature_s=None) -> None:
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.temperature_s = temperature_s if temperature_s is not None else temperature
        self.c_entropy = torch.nn.CrossEntropyLoss()

    @staticmethod
    def mask_diagonal(similarities: torch.Tensor) -> torch.Tensor:
        return similarities[~torch.eye(similarities.shape[0], dtype=bool)].reshape(similarities.shape[0], -1)

    def get_teacher_distribution(self, teacher_similarities: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(self.mask_diagonal(teacher_similarities) / self.temperature, dim=-1)

    def cross_entropy_loss(
        self, teacher_similarities: torch.Tensor, student_similarities: torch.Tensor
    ) -> torch.Tensor:
        p = self.get_teacher_distribution(teacher_similarities)
        q_unnormalized = self.mask_diagonal(student_similarities) / self.temperature_s
        return self.c_entropy(q_unnormalized, p)

    def forward(self, teacher_features: torch.Tensor, student_features: torch.Tensor) -> torch.Tensor:
        normalized_teacher_features = torch.nn.functional.normalize(teacher_features, dim=1)
        normalized_student_features = torch.nn.functional.normalize(student_features, dim=1)
        teacher_similarities = normalized_teacher_features @ normalized_teacher_features.T
        student_similarities = normalized_student_features @ normalized_student_features.T
        return self.cross_entropy_loss(teacher_similarities, student_similarities)


def get_loss(loss_type: str, batch_size: Optional[int] = None, device: Optional[str] = None):
    if loss_type == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_type == "mae":
        criterion = torch.nn.L1Loss()
    elif loss_type == "cosine_distance":
        criterion = CosineDistanceLoss()
    elif "combined" in loss_type:
        # Parse losses from losstype, expect: combinedFLOAT__loss1__loss2
        pattern = r"^combined(\d+(?:\.\d+)?)(?:__(.+))?$"
        match = re.match(pattern, loss_type)
        alpha = float(match.group(1))
        loss_1 = match.group(2).split("__")[0]
        loss_2 = match.group(2).split("__")[1]
        criterion_1 = get_loss(loss_1, batch_size, device)
        criterion_2 = get_loss(loss_2, batch_size, device)
        criterion = CombinedLoss(criterion_1, criterion_2, alpha=alpha)

    elif "glocal" in loss_type:
        # Parse temperature from loss type, expect 1 or two temperatures
        pattern = r"^glocal_(\d+(?:\.\d+)?)(?:_(\d+(?:\.\d+)?))?$"
        match = re.match(pattern, loss_type)
        temperature_t = float(match.group(1))
        temperature_s = (
            float(match.group(2)) if match.group(2) is not None else None
        )  # Will be None if the second number part didn't match
        criterion = ContrastiveLoss(temperature=temperature_t, temperature_s=temperature_s)
    elif "cka" in loss_type:
        criterion = CKALoss(batch_size=batch_size, device=device, type=loss_type.replace("cka_", ""))
    else:
        raise ValueError(f"Loss function {loss_type} not supported.")
    return criterion
