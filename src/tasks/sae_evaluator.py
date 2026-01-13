import argparse
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loader import get_feature_dl
from src.data.data_utils import create_train_val_loaders
from src.eval.metrics import RepresentationMetrics
from src.models.sae import TopKSparseAutoencoder
from src.tasks.probe_evaluator import SingleModelEvaluator
from src.utils.loss_utils import Regularizer, get_loss
from src.utils.utils import (
    as_list,
    check_feature_existence,
    check_single_instance,
)
from src.utils.wandb_logger import WandBLogger


@dataclass
class OptimConfig:
    initial_lr: List[float] | float = 3e-4
    final_lr: float = 1e-6
    warmup_epochs: int = 5
    grad_clip: float = 5
    reg_lambda: float = 0.2
    regularization: str = "weight_decay"


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_workers: int = 4
    epochs: List[int] | int = 100
    rep_loss: str = "mae"
    sae_k: int | None = None
    hidden_dim: int | None = None
    sae_increase_factor: float | None = 8


@dataclass
class ValidationConfig:
    patience: int = 10
    val_proportion: float = 0.1
    min_delta: float = 1e-5
    eval_with_lin_probe: bool = True


class SAEEvaluator:
    def __init__(
        self,
        train_config: TrainingConfig,
        optim_config: OptimConfig,
        val_config: ValidationConfig | None = None,
        wandb_logger: WandBLogger | None = None,
        device: str = "cuda",
        seed: int = 42,
        normalize: bool = True,
    ):
        self.seed = seed
        self.train_config = train_config
        self.optim_config = optim_config
        self.val_config = val_config or ValidationConfig()
        self.wandb_logger = wandb_logger or self._default_wandb_logger()
        self.normalize = normalize
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU instead")
            device = "cpu"
        self.device = device

        self.loss_fn = get_loss(
            self.train_config.rep_loss,
            batch_size=self.train_config.batch_size,
            device=self.device,
        )
        self.model: TopKSparseAutoencoder | None = None
        # TODO: Might need to move this to training as it might be changed during optimization
        self.reg = Regularizer(self.optim_config.regularization, self.optim_config.reg_lambda)
        self.rep_metrics = RepresentationMetrics(
            max_samples_cka=20000,
            contrastive_temperature=1.0,
            device=self.device,
        )

    def _default_wandb_logger(self) -> WandBLogger:
        return WandBLogger(
            project="rep2rep",
            enabled=True,
            config={
                "learning_rates": self.optim_config.initial_lr,
                "reg_lambda": self.optim_config.reg_lambda,
                "epochs": self.train_config.epochs,
                "val_proportion": self.val_config.val_proportion,
                "patience": self.val_config.patience,
                "min_delta": self.val_config.min_delta,
            },
        )

    def evaluate(
        self,
        feature_dir: str,
        model_dir: str,
        latent_feature_dir: str,
        extract_train: bool = False,
        extract_test: bool = True,
    ):
        """Train the SAE and save the latents.

        Args:
            feature_dir: Directory containing the features of dataset for specific model.
            model_dir: Directory to save the SAE model.
            latent_feature_dir: Directory to save the latents.
            extract_train: Whether to extract the latents from the training set.
            extract_test: Whether to extract the latents from the test set.
        """
        logger.info(f"Training SAE and extracting latents for {feature_dir}")
        feature_dir = check_single_instance(feature_dir, "feature_dir")
        model_dir = check_single_instance(model_dir, "model_dir")
        latent_feature_dir = check_single_instance(latent_feature_dir, "latent_feature_dir")
        feature_available = check_feature_existence(feature_dir, check_train=True)
        if not feature_available:
            raise ValueError(
                "Features do not exist. Please run feature extraction first, i.e., `task=feature_extraction`"
            )

        feature_train_loader, feature_test_loader = get_feature_dl(
            feature_dir=feature_dir,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            fewshot_k=-1,
            load_train=True,
            normalize=self.normalize,
        )

        # Optimize hyperparameters -> sets best hyperparameters in the end
        logger.info("Optimizing learning rate and number of epochs")
        self.optimize_hyperparams(feature_train_loader)

        # Retrain SAE with best hyperparameters
        logger.info("Training SAE")
        train_loss, val_loss, epoch, filename = self.train(
            train_loader=feature_train_loader,
            model_dir=model_dir,
            with_val=False,
            store_model=True,
        )

        # Extract latents
        metric_dict = {
            "learning_rate": self.optim_config.initial_lr,
            "epochs": self.train_config.epochs,
        }
        metric_dict.update(self.model.get_config())

        if extract_train:
            train_loss, train_latents, train_targets = self.infer(feature_train_loader, get_latents_n_targets=True)
            self._save_latents(train_latents, train_targets, latent_feature_dir, "train")

        metric_dict["train_recon_loss"] = train_loss

        if extract_test:
            test_loss, test_latents, test_targets = self.infer(feature_test_loader, get_latents_n_targets=True)
            self._save_latents(test_latents, test_targets, latent_feature_dir, "test")
            metric_dict["test_recon_loss"] = test_loss

        if self.val_config.eval_with_lin_probe and filename is not None and os.path.exists(filename):
            logger.info("Evaluating linear probe on transformed features.")
            linear_probe_path = os.path.dirname(filename)
            evaluator = SingleModelEvaluator(
                batch_size=self.train_config.batch_size,
                num_workers=self.train_config.num_workers,
                lrs=self.optim_config.initial_lr,
                epochs=self.train_config.epochs,
                seed=self.seed,
                device=self.device,
                fewshot_k=-1,
                feature_dirs=feature_dir,
                model_dirs=[linear_probe_path],
                results_dir=linear_probe_path,
                val_proportion=self.val_config.val_proportion,
                reg_lambda=self.optim_config.reg_lambda,
                regularization=self.optim_config.regularization,
                force_train=False,
                model_fn="probe.pkl",
                premodel_filename=filename,
            )
            probe_metrics = evaluator.evaluate()
            metric_dict["linear_probe_eval"] = probe_metrics

        return metric_dict

    def train(
        self,
        train_loader: DataLoader,
        model_dir: Optional[str] = None,
        with_val: bool = True,
        store_model: bool = True,
    ) -> Tuple[float, float, int]:
        """Train the SAE. Uses early stopping if validation is enabled (val_proportion > 0 and with_val=True)."""

        self.train_config.epochs = check_single_instance(self.train_config.epochs, "epochs")

        optimizer, scheduler, tmp_train_loader, tmp_val_loader = self._init_training(train_loader, with_val=with_val)

        len_loader = len(tmp_train_loader)
        logger.info(
            f"\nTraining SAE for {self.train_config.epochs=} with dataloader containing {len_loader=} batches. "
            f"\nWe use the following optimizer and training config:\n{self.optim_config=}\n{self.train_config=}\n",
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        for epoch in range(self.train_config.epochs):
            # Training loop
            train_loss, current_lr = self._train_epoch(tmp_train_loader, optimizer, scheduler, epoch, len_loader)
            # Validation loop
            if tmp_val_loader is not None and self.val_config.val_proportion > 0:
                val_loss = self.infer(tmp_val_loader, get_latents_n_targets=False)

                # Early stopping check
                if val_loss < best_val_loss - self.val_config.min_delta:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.val_config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        if best_model_state is None:
                            raise ValueError("No best model state found")
                        self.model.load_state_dict(best_model_state)
                        break
            logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.5f}, "
                f"Best Val Loss = {best_val_loss:.5f}, LR = {current_lr:.3e}"
            )
        if store_model and model_dir is not None:
            if not os.path.exists(model_dir):
                Path(model_dir).mkdir(parents=True, exist_ok=True)
            filename = os.path.join(model_dir, "sae.pt")
            torch.save(self.model.state_dict(), filename)
            logger.info(f"SAE trained and saved to {filename}")
        else:
            logger.info("SAE not saved")
            filename = None

        return train_loss, best_val_loss, epoch, filename

    def infer(
        self, dataloader: DataLoader, get_latents_n_targets: bool = True
    ) -> float | Tuple[float, torch.Tensor, torch.Tensor]:
        loss = 0.0
        latents, targets = [], []
        pbar = tqdm(
            dataloader,
            desc="Inference",
            disable=logger.getEffectiveLevel() > logging.INFO,
        )
        cnt = 0
        with torch.no_grad():
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                recon, latent = self.model(x)

                loss += self.loss_fn(recon, x) + self.reg.reg_loss(self.model)

                if get_latents_n_targets:
                    latents.append(latent.cpu())
                    targets.append(y.cpu())
                cnt += 1
                pbar.set_postfix({"loss": f"{loss / cnt:.5f}"})
        loss /= len(dataloader)
        if get_latents_n_targets:
            latent = torch.cat(latents)
            target = torch.cat(targets)
            return loss, latent, target
        else:
            return loss

    def optimize_hyperparams(self, train_loader: DataLoader):
        """Optimize the hyperparameters of the SAE."""
        best_val_loss = float("inf")
        best_lr = None
        best_epochs = None
        logger.info("Optimizing learning rate and number of epochs")
        for lr in as_list(self.optim_config.initial_lr):
            self.optim_config.initial_lr = lr
            train_loss, val_loss, epoch, _ = self.train(
                train_loader=train_loader,
                with_val=True,
                store_model=False,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr
                best_epochs = epoch

        logger.info(
            f"After tuning the learning rate, the best val loss is {best_val_loss} with {best_epochs=} and {best_lr=}."
        )
        logger.info(f"Setting the learning rate to {best_lr} and the number of epochs to {best_epochs}.")
        self.optim_config.initial_lr = best_lr
        self.train_config.epochs = best_epochs

    def _get_lr_scheduler(self, optimizer, len_loader: int):
        """Get the learning rate scheduler."""

        def _lr_adjuster(step):
            warmup_steps = self.optim_config.warmup_epochs * len_loader
            if step < warmup_steps:
                lr = (step / warmup_steps) * self.optim_config.initial_lr
            else:
                total_steps = self.train_config.epochs * len_loader
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                lr = self.optim_config.final_lr + 0.5 * (self.optim_config.initial_lr - self.optim_config.final_lr) * (
                    1 + np.cos(np.pi * progress)
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        return _lr_adjuster

    def _init_training(self, train_loader: DataLoader, with_val: bool = True):
        """Initialize optimizer & scheduler and potential train/val loaders."""
        sample_batch, _ = next(iter(train_loader))

        self.model = TopKSparseAutoencoder(
            input_dim=sample_batch.shape[1],  # input dimension from first batch
            hidden_dim=self.train_config.hidden_dim,
            k=self.train_config.sae_k,
            increase_factor=self.train_config.sae_increase_factor,
        )
        self.model = self.model.to(self.device)

        if self.val_config.val_proportion > 0 and with_val:
            tmp_train_loader, tmp_val_loader = create_train_val_loaders(
                train_loader,
                val_proportion=self.val_config.val_proportion,
                seed=self.seed,
            )
        else:
            tmp_train_loader = train_loader
            tmp_val_loader = None

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.optim_config.initial_lr,
            weight_decay=self.optim_config.reg_lambda,
        )
        scheduler = self._get_lr_scheduler(optimizer, len(tmp_train_loader))

        return optimizer, scheduler, tmp_train_loader, tmp_val_loader

    def _train_step(self, x, optimizer):
        """Execute single training step."""
        x = x.to(self.device)
        optimizer.zero_grad(set_to_none=True)
        recon, _ = self.model(x)
        loss = self.loss_fn(recon, x) + self.reg.reg_loss(self.model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.optim_config.grad_clip)
        optimizer.step()

        return loss

    def _train_epoch(self, loader, optimizer, scheduler, epoch: int, len_loader: int):
        """Run one epoch of training."""
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{self.train_config.epochs}",
            disable=logger.getEffectiveLevel() > logging.INFO,
        )
        for i, (x, _) in enumerate(pbar):
            step = epoch * len_loader + i
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler(step)
            loss = self._train_step(x, optimizer)
            train_loss += loss
            avg_loss = train_loss / (i + 1)
            pbar.set_postfix({"train loss": f"{avg_loss:.5f}", "lr": f"{current_lr:.5e}"})

        return train_loss / len_loader, current_lr

    def _save_latents(
        self,
        latents: torch.Tensor,
        targets: torch.Tensor,
        latent_feature_dir: str,
        suffix: str,
    ):
        np_latents = latents.cpu().numpy()
        np_targets = targets.cpu().numpy()
        np.savez_compressed(
            os.path.join(latent_feature_dir, f"latents_features_{suffix}.npz"),
            data=np_latents,
        )
        np.savez_compressed(os.path.join(latent_feature_dir, f"targets_{suffix}.npz"), data=np_targets)


def get_sae_config_from_args(
    args: argparse.Namespace,
) -> dict[str, Union[TrainingConfig, OptimConfig, ValidationConfig, WandBLogger]]:
    """Get the SAE configuration from the command line arguments."""
    args_dict = vars(args)

    def update_config(cls):
        config_defaults = asdict(cls())
        config_args = {k: args_dict[k] for k in config_defaults if k in args_dict}
        return cls(**{**config_defaults, **config_args})

    configs = {}
    configs["train_config"] = update_config(TrainingConfig)
    configs["optim_config"] = update_config(OptimConfig)
    configs["val_config"] = update_config(ValidationConfig)

    configs["wandb_logger"] = WandBLogger(
        project=args_dict.get("project", "rep2rep"),
        config={
            "learning_rates": configs["optim_config"].initial_lr,
            "epochs": configs["train_config"].epochs,
            "val_proportion": configs["val_config"].val_proportion,
            "patience": configs["val_config"].patience,
            "min_delta": configs["val_config"].min_delta,
        },
        enabled=args_dict.get("use_wandb", False),
    )

    configs["device"] = args.device
    configs["seed"] = args.seed
    configs["normalize"] = args.normalize

    return configs
