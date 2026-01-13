import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from timm.models.layers import drop_path
from torch.nn.modules.batchnorm import _NormBase
from torch.utils.data import DataLoader

from src.models.thingsvision import ThingsvisionModel
from src.tasks.linear_probe import LinearProbe

# ---- Acknowledgement ----
# This code has been mainly taken from the following repository:
# https://github.com/Atten4Vis/CAE/blob/a7fd1628176358e3e76b0b042b0f6e9f3cd7c76c/models/modeling_finetune.py
#
# Citation:
# @Article{ContextAutoencoder2022,
# title={Context Autoencoder for Self-Supervised Representation Learning},
# author={Chen, Xiaokang and Ding, Mingyu and Wang, Xiaodi and Xin, Ying and Mo, Shentong
# and Wang, Yunhao and Han, Shumin and Luo, Ping and Zeng, Gang and Wang, Jingdong},
# journal={arXiv preprint arXiv:2202.03026},
# year={2022}
# }

# ---- Attentive Block ----


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        # NOTE: original code uses head_dim = (dim // num_heads), but we doubled
        # the head dimension based on the results of the ablation study
        head_dim = (dim // num_heads) * 2
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.record_attention = False  # Flag to record attention weights
        self.attn_rec = []  # To store the last attention weights

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        if self.record_attention:
            self.attn_rec.append(attn.detach().cpu().numpy())
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class AttentiveBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()

        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k):
        x_q = self.norm_q(x_q + pos_q)
        x_k = self.norm_k(x_kv + pos_k)
        x_v = self.norm_v(x_kv)

        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class LP_BatchNorm(_NormBase):
    """A variant used in linear probing.
    To freeze parameters (normalization operator specifically), model set to eval mode during linear probing.
    According to paper, an extra BN is used on the top of encoder to calibrate the feature magnitudes.
    In addition to self.training, we set another flag in this implement to control BN's behavior to train in eval mode.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(LP_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(input.dim()))

    def forward(self, input, is_train):
        """
        We use is_train instead of self.training.
        """
        self._check_input_dim(input)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # if self.training and self.track_running_stats:
        if is_train and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if is_train:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not is_train or self.track_running_stats else None,
            self.running_var if not is_train or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


# ---- Main Attentive Probe Model ----


class AttentiveProbeModel(nn.Module):
    def __init__(
        self,
        dim: int | list[int] | torch.Size,
        num_heads: int,
        num_classes: int,
        attention_dropout: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__()
        drop, attn_drop = attention_dropout
        self.attn_block = AttentiveBlock(dim, num_heads, drop=drop, attn_drop=attn_drop)
        # self.bn = LP_BatchNorm(dim, affine=False)
        self.bn = nn.BatchNorm1d(
            dim,
            eps=1e-05,
            momentum=0.1,
            # default is affine=True of nn.BatchNorm1d, but it was set to False in the original code cited above
            # after ablation study, it was found that setting affine=True gives better performance, see branch BatchNormAffineTrueFalse
            affine=True,
            track_running_stats=True,
        )
        self.classifier = nn.Linear(dim, num_classes)
        self.query_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query_token, std=0.02)

    def forward(self, x, return_feature_repr=False):
        # x shape: (B, N, D) where N is number of patches and D is feature dimension
        batch_size = x.size(0)
        query_tokens = self.query_token.expand(batch_size, -1, -1)
        query_tokens = self.attn_block(query_tokens, x, 0, 0)
        feature_repr = self.bn(query_tokens[:, 0, :])
        # feature_repr = query_tokens[:, 0, :]
        x = self.classifier(feature_repr)
        if return_feature_repr:
            return x, feature_repr
        else:
            return x


# ---- Dimension Alignment ----


class DimensionAlignment(nn.Module):
    """
    Align the dimensions of the input features to a shared dimension.
    """

    def __init__(
        self,
        model_dims: list[int],
        dim_alignment: str = "zero_padding",
        shared_dim: int | None = None,
        always_project: bool = False,
    ):
        """
        Args:
            model_dims: list of dimensions of the input features of each model
            dim_alignment: "zero_padding" or "linear_projection"
            shared_dim: shared dimension of the input features
            always_project: if True, always project the input features to the shared dimension, even
            if the shared dimension is equal to the dimension of the input features.
        """
        super().__init__()
        self.dim_alignment = dim_alignment
        self.model_dims = model_dims
        self.n_models = len(model_dims)
        self.always_project = always_project

        if self.n_models < 2:
            raise ValueError("n_models must be greater than 1")

        self.shared_dim = shared_dim or max(model_dims)

        if dim_alignment == "linear_projection":
            self.linear_projections = nn.ModuleList([self._get_layer_to_project(dim) for dim in model_dims])
        elif dim_alignment == "zero_padding" and self.shared_dim < min(model_dims):
            raise ValueError("shared_dim must be greater than or equal to the minimum of model_dims")

    def _get_layer_to_project(self, dim):
        if self.always_project:
            return nn.Linear(dim, self.shared_dim)
        else:
            if dim == self.shared_dim:
                return nn.Identity()
            else:
                return nn.Linear(dim, self.shared_dim)

    def _zero_padding(self, x):
        """
        Pad the input features to the shared dimension.
        """
        x = [F.pad(x[i], (0, self.shared_dim - x[i].shape[1], 0, 0)) for i in range(len(x))]
        x = torch.stack(x, dim=1)
        return x

    def _linear_projection(self, x):
        """
        Project the input features with a linear projection to the shared dimension.
        """
        x_aligned = []
        for i, lin_proj in enumerate(self.linear_projections):
            x_aligned.append(lin_proj(x[i]))
        x_aligned = torch.stack(x_aligned, dim=1)
        return x_aligned

    def forward(self, x):
        if self.dim_alignment == "zero_padding":
            return self._zero_padding(x)
        elif self.dim_alignment == "linear_projection":
            return self._linear_projection(x)
        else:
            raise ValueError(
                f"Unknown dimension alignment: {self.dim_alignment}. Available options are: zero_padding, linear_projection."
            )


# ---- Attentive Probe Task ----


class AttentiveProbe(LinearProbe):
    """Attentive probe for downstream task evaluation."""

    def __init__(
        self,
        dim: int | list[int] | torch.Size,
        num_heads: int,
        reg_lambda: float,
        lr: float,
        epochs: int,
        device: str,
        seed: int,
        dimension_alignment: str = "zero_padding",
        always_project: bool = False,
        logit_filter: torch.Tensor | None = None,
        regularization: str = "weight_decay",
        use_data_parallel: bool = False,
        filename: str | None = None,
        force_train: bool = False,
        premodel: str | torch.nn.Module | ThingsvisionModel | None = None,
        freeze_premodel: bool = True,
        use_class_weights: bool = True,
        min_learning_steps: int = 1000,
        grad_norm_clip: float | None = None,
        attention_dropout: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__(
            reg_lambda=reg_lambda,
            lr=lr,
            epochs=epochs,
            device=device,
            seed=seed,
            logit_filter=logit_filter,
            regularization=regularization,
            use_data_parallel=use_data_parallel,
            filename=filename,
            force_train=force_train,
            premodel=premodel,
            freeze_premodel=freeze_premodel,
            use_class_weights=use_class_weights,
            min_learning_steps=min_learning_steps,
            grad_norm_clip=grad_norm_clip,
        )

        self.dim = dim
        self.num_heads = num_heads
        self.dimension_alignment = dimension_alignment
        self.attention_dropout = attention_dropout
        logger.warning(
            "DeprecationWarning: Dimension alignment is deprecated and not used anymore. "
            f"Dimension alignment argument ({self.dimension_alignment}) is ignored and zero_padding "
            "is handeled by the dataset (StackedZeroPadFeatureCombiner). "
            "NOTE: linear_projection is not supported at the moment."
        )
        self.always_project = always_project

    def _init_new_model(
        self,
        input_shape: int| list[int],
        output_shape: int,
        premodel: torch.nn.Module | None = None,
    ) -> torch.nn.Module | torch.nn.DataParallel:
        
        # if isinstance(input_shape, list):
        #     input_shape = input_shape[0]

        # if input_shape != self.dim:
        #     raise ValueError(f"Input feature dimension {input_shape} does not match the expected dimension {self.dim}")

        model = AttentiveProbeModel(self.dim, self.num_heads, output_shape, self.attention_dropout)

        if premodel is not None:
            model = torch.nn.Sequential(premodel, model)

        model = model.to(self.device)
        if self.use_data_parallel and torch.cuda.is_available():
            model = torch.nn.DataParallel(
                model,
                device_ids=list(range(torch.cuda.device_count())),
            )
        return model
