import torch
import torch.nn as nn


class TopKSparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        k: int | None = None,
        hidden_dim: int | None = None,
        increase_factor: float | None = None,
    ):
        """
        Initialize the TopK Sparse Autoencoder.

        Args:
            input_dim: The size of the input feature vector.
            k: The count of active features to retain (top-k). If not specified, it defaults to None,
                resulting in the hidden dimension being set to half of the input dimension.
            hidden_dim: The number of units in the hidden layer or the number of dictionary elements.
            increase_factor: A multiplier that determines how much to scale the input dimension when
                defining the hidden dimension. If specified, `hidden_dim` will be disregarded; otherwise,
                it will serve as the starting point for the hidden dimension.
        """
        super().__init__()

        if increase_factor is not None and increase_factor >= 1.0:
            self.hidden_dim = int(input_dim * increase_factor)
            self.increase_factor = increase_factor
        elif hidden_dim is not None and hidden_dim >= 1:
            self.hidden_dim = hidden_dim
            self.increase_factor = hidden_dim / input_dim
        else:
            raise ValueError(
                "Either hidden_dim or increase_factor (>=1) must be provided"
            )

        if k is None:
            k = input_dim // 2

        if k > self.hidden_dim:
            raise ValueError(
                f"k cannot be greater than hidden_dim: {k=} > {self.hidden_dim=}"
            )

        self.k = k
        self.input_dim = input_dim
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.latent_bias = nn.Parameter(torch.zeros(self.hidden_dim))

    def get_topk_mask(self, h: torch.Tensor) -> torch.Tensor:
        """Get binary mask for top-k activations."""
        top_k_ids = torch.topk(h, self.k, dim=1)[1]
        mask = torch.zeros_like(h)
        mask.scatter_(1, top_k_ids, 1)
        return mask

    def forward(
        self, x: torch.Tensor, return_recon: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        latents = relu(topk(encoder(x - pre_bias) + latent_bias))
        recon = decoder(latents) + pre_bias
        """
        x_centered = x - self.pre_bias

        h = self.encoder(x_centered) + self.latent_bias

        mask = self.get_topk_mask(h)
        h_sparse = h * mask

        h_sparse = torch.relu(h_sparse)

        if not return_recon:
            return h_sparse

        recon = self.decoder(h_sparse) + self.pre_bias

        return recon, h_sparse

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "k": self.k,
            "hidden_dim": self.hidden_dim,
            "increase_factor": self.increase_factor,
        }
