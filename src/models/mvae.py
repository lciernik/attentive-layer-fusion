import abc
import math
import re

import torch
from torch.types import _size

DistrType = torch.distributions.Distribution


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


class SimpleMVAEModel(torch.nn.Module):
    def __init__(
        self,
        x_shape: int,
        y_shape: int,
        device: str,
        method="poe",
        distribution="laplace",
        linear=True,
        normalize=False,
        individual_variance=False,
    ):
        super(SimpleMVAEModel, self).__init__()
        self.latent_shape = max(x_shape, y_shape)
        self.device = device
        self.normalize = normalize

        assert method in ["moe", "poe"], "VAE combination method must be 'moe' or 'poe'"
        self.method = method

        self.encoders, self.decoders = self._get_encoders_n_decoders(
            linear, x_shape, y_shape
        )

        self.var_predictor = self._get_var_predictor(
            individual_variance, x_shape, y_shape
        )

        if distribution == "laplace":
            self.distribution = torch.distributions.Laplace
        elif distribution == "normal":
            self.distribution = torch.distributions.Normal
        else:
            raise ValueError(
                f"Distribution {distribution} must be 'laplace' or 'normal'"
            )

    def _get_encoders_n_decoders(
        self,
        linear: bool,
        x_shape: int,
        y_shape: int,
    ) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
        """Create encoders and decoders for the MVAE model."""

        def create_network_part(input_shape: int, output_shape: int) -> torch.nn.Module:
            if linear:
                return torch.nn.Linear(input_shape, output_shape)
            return torch.nn.Sequential(
                torch.nn.Linear(input_shape, input_shape),
                torch.nn.SELU(),
                torch.nn.Linear(input_shape, output_shape),
            )

        encoders = torch.nn.ModuleList(
            [
                create_network_part(x_shape, self.latent_shape),
                create_network_part(y_shape, self.latent_shape),
            ]
        )
        decoders = torch.nn.ModuleList(
            [
                create_network_part(self.latent_shape, x_shape),
                create_network_part(self.latent_shape, y_shape),
            ]
        )
        return encoders.to(self.device), decoders.to(self.device)

    def _get_var_predictor(
        self, individual_variance: bool, x_shape: int, y_shape: int
    ) -> torch.nn.ModuleList:
        """Create variance predictors for the MVAE model.

        Args:
            individual_variance: Whether to predict individual variances for each dimension.
            x_shape: Dimension of input x.
            y_shape: Dimension of input y.

        Returns:
            ModuleList containing variance predictors for x and y.
        """

        def create_var_predictor(input_shape: int) -> torch.nn.Sequential:
            if individual_variance:
                return torch.nn.Sequential(
                    torch.nn.Linear(input_shape, input_shape),
                    torch.nn.SELU(),  # Match encoder/decoder activation
                    torch.nn.Linear(input_shape, self.latent_shape),
                    torch.nn.Softplus(),
                )
            return torch.nn.Sequential(
                torch.nn.Linear(input_shape, input_shape // 2),
                torch.nn.SELU(),  # Match encoder/decoder activation
                torch.nn.Linear(input_shape // 2, 1),
                torch.nn.Softplus(),
            )

        return torch.nn.ModuleList(
            [
                create_var_predictor(x_shape),
                create_var_predictor(y_shape),
            ]
        ).to(self.device)

    def forward(self, xy):
        x, y = xy
        # Encode
        q1, q2 = self.encode(x, y)
        # Fusion step (PoE or MoE)
        q_joint = self.combine_representations(q1, q2)
        if self.training:
            # Sample from the joint distribution
            z_joint = q_joint.rsample()
        else:
            # Take the mean of the joint distribution
            z_joint = q_joint.mean
        return z_joint

    def encode(self, x, y):
        mu1 = self.encoders[0](x)
        mu2 = self.encoders[1](y)
        s1 = torch.clamp(self.var_predictor[0](x), min=1e-5)
        s2 = torch.clamp(self.var_predictor[1](y), min=1e-5)
        return self.distribution(mu1, s1), self.distribution(mu2, s2)

    def decode(self, z):
        x_recon = self.decoders[0](z)
        y_recon = self.decoders[1](z)
        if self.normalize:
            x_recon = torch.nn.functional.normalize(x_recon, p=2, dim=-1)
            y_recon = torch.nn.functional.normalize(y_recon, p=2, dim=-1)
        return x_recon, y_recon

    def combine_representations(self, q1, q2):
        # Fusion step (PoE or MoE)
        if self.method == "poe":
            # Combine precisions
            if self.distribution == torch.distributions.Laplace:
                precision1 = 1.0 / (2 * q1.scale**2)
                precision2 = 1.0 / (2 * q2.scale**2)
                combined_precision = (
                    precision1 + precision2 + 0.5
                )  # Currently using Laplace(0,1) as prior
                scale_poe = torch.sqrt(1.0 / (2 * combined_precision))
            elif self.distribution == torch.distributions.Normal:
                precision1 = 1.0 / (q1.scale**2)
                precision2 = 1.0 / (q2.scale**2)
                combined_precision = (
                    precision1 + precision2 + 1
                )  # Currently using Normal(0,1) as prior
                scale_poe = torch.sqrt(1.0 / combined_precision)
            else:
                raise ValueError("Distribution must be 'laplace' or 'normal'")
            mu_poe = (q1.mean * precision1 + q2.mean * precision2) / combined_precision
            q_joint = self.distribution(mu_poe, scale_poe)
        elif self.method == "moe":
            q_joint = SimpleMOE(q1, q2)
            # Sample from q1 or q2 with equal prob
        else:
            raise ValueError("Unknown fusion method")
        return q_joint


class SimpleMOE(torch.distributions.Distribution):
    arg_constraints = {}

    def __init__(
        self, q1: torch.distributions.Distribution, q2: torch.distributions.Distribution
    ):
        super().__init__(batch_shape=q1.batch_shape)
        self.q1 = q1
        self.q2 = q2

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        z1 = self.q1.rsample(sample_shape)  # [K, B, D]
        z2 = self.q2.rsample(sample_shape)  # [K, B, D]
        return torch.where(torch.rand_like(z1) > 0.5, z1, z2)  # K samples from mixture

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_q1 = self.q1.log_prob(value)
        log_q2 = self.q2.log_prob(value)
        log_q = torch.stack([log_q1, log_q2], 0)
        return log_mean_exp(log_q, dim=0)

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        mean_cond_var = 0.5 * (self.q1.variance + self.q2.variance)
        var_cond_mean = 0.25 * (
            self.q1.mean**2 + self.q2.mean**2 - 2 * self.q1.mean * self.q2.mean
        )
        return mean_cond_var + var_cond_mean

    @property
    def scale(self):
        return self.variance**0.5

    @property
    def stddev(self):
        return self.variance**0.5

    @property
    def mean(self):
        return 0.5 * (self.q1.mean + self.q2.mean)

    @property
    def size(self):
        return self.q1.mean.size

    @property
    def shape(self):
        return self.q1.mean.shape


class MVAELoss(torch.nn.Module, abc.ABC):
    def __init__(
        self,
        method: str = "moe",
        loose: bool = False,
        distribution: str = "Laplace",
        loss_modification: str = "",
    ):
        """
        MVAE loss for multimodal VAEs.
        Inspired by https://github.com/iffsid/mmvae/
        Shi, Yuge, Brooks Paige, and Philip Torr.
        "Variational mixture-of-experts autoencoders for multi-modal deep generative models."
        Advances in neural information processing systems 32 (2019).

        Args:
            method (str): "moe" or "poe"
            loose (bool): Use loose bound (average over modalities outside the log)
            distribution (torch.distributions.Distribution): Distribution to use for the latent variables
            loss_modification (str): Modification to the loss function. Options are "mvae" or "mvaeelbo" with "scaleX", "normalize", "NL"
        """
        super().__init__()
        assert method in ["moe", "poe"], "method must be 'moe' or 'poe'"
        self.method = method
        self.loose = loose

        match = re.search(r"K(\d+)", loss_modification)
        if match and match.group(1):
            self.K = int(match.group(1))
        else:
            self.K = self._default_k

        if distribution == "laplace":
            self.distribution = torch.distributions.Laplace
        elif distribution == "normal":
            self.distribution = torch.distributions.Normal
        else:
            raise ValueError(
                f"Distribution {distribution} must be 'laplace' or 'normal'"
            )
        self.prior = self.distribution(0, 1)  # isotropic prior
        match = re.search(r"scale(\d+(?:\.\d+)?)", loss_modification)
        if match:
            self.beta = float(match.group(1))
        else:
            self.beta = 0.1  # We always scale, maybe cleaner to make it explicit?
        self.loss_modification = loss_modification

    @abc.abstractmethod
    def _get_detached_distributions(
        self, q1, q2, q_joint, model
    ) -> tuple[DistrType, DistrType, DistrType]:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_kl(self, z, q):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_final_loss(self, log_w, z_joint):
        raise NotImplementedError

    @property
    def _default_k(self) -> int:
        return 1

    def forward(self, x, y, model):
        # Likelihood scaling
        ls_1 = max(x.shape[-1], y.shape[-1]) / x.shape[-1]
        ls_2 = max(x.shape[-1], y.shape[-1]) / y.shape[-1]

        # Encode
        q1, q2 = model.encode(x, y)

        # Fusion step (PoE or MoE)
        q_joint = model.combine_representations(q1, q2)

        q_1_detached, q_2_detached, q_joint_detached = self._get_detached_distributions(
            q1, q2, q_joint, model
        )

        z_joint = q_joint.rsample((self.K,))  # [K, B, D]
        if self.loose:
            z1 = q1.rsample((self.K,))  # [K, B, D]
            z2 = q2.rsample((self.K,))  # [K, B, D]
            combinations = [
                (q_1_detached, z1),
                (q_2_detached, z2),
                (q_joint_detached, z_joint),
            ]
        else:
            combinations = [(q_joint_detached, z_joint)]

        log_ws = []
        for q, z in combinations:
            x_recon, y_recon = model.decode(z)  # [K, B, Dx], [K, B, Dy]y

            log_px_z = ls_1 * self.distribution(x_recon, 0.2).log_prob(
                x.unsqueeze(0)
            ).sum(-1)
            log_py_z = ls_2 * self.distribution(y_recon, 0.2).log_prob(
                y.unsqueeze(0)
            ).sum(-1)

            kl = self._get_kl(z, q)

            log_w = log_px_z + log_py_z - kl
            log_ws.append(log_w)

        log_w = torch.stack(log_ws).mean(0)  # average before DReG weighting

        return self._get_final_loss(log_w, z_joint)


class MVAEElboLoss(MVAELoss):
    def _get_detached_distributions(
        self,
        q1,
        q2,
        q_joint,
        model,
    ) -> tuple[DistrType, DistrType, DistrType]:
        return q1, q2, q_joint

    def _get_kl(self, z, q):
        return self.beta * kl_divergence(q, self.prior).sum(-1).unsqueeze(0)

    def _get_final_loss(self, log_w, z_joint):
        return -1 * log_w.mean()


class MVAEDReGLoss(MVAELoss):
    @property
    def _default_k(self) -> int:
        return 5

    def _get_detached_distributions(
        self,
        q1,
        q2,
        q_joint,
        model,
    ) -> tuple[DistrType, DistrType, DistrType]:
        q_1_detached = self.distribution(q1.loc.detach(), q1.scale.detach())
        q_2_detached = self.distribution(q2.loc.detach(), q2.scale.detach())
        q_joint_detached = model.combine_representations(q_1_detached, q_2_detached)
        return q_1_detached, q_2_detached, q_joint_detached

    def _get_kl(self, z, q):
        lpz = self.prior.log_prob(z).sum(-1)
        lqz = q.log_prob(z).sum(-1)
        kl = self.beta * (lqz - lpz)
        return kl

    def _get_final_loss(self, log_w, z_joint):
        with torch.no_grad():
            grad_wt = (log_w - torch.logsumexp(log_w, 0, keepdim=True)).exp()

        if z_joint.requires_grad:
            # Don't register Hook in eval mode
            z_joint.register_hook(lambda g: grad_wt.unsqueeze(-1) * g)

        return -(grad_wt * log_w).mean()


def get_mvae_loss(loss_modification: str, **kwargs) -> MVAELoss:
    if "elbo" in loss_modification:
        class_ = MVAEElboLoss
    elif "dreg" in loss_modification:
        class_ = MVAEDReGLoss
    else:
        raise ValueError(f"Unknown loss modification: {loss_modification}")

    return class_(loss_modification=loss_modification, **kwargs)


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)
