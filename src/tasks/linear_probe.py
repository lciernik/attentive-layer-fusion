from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.models.thingsvision import ThingsvisionModel
from src.utils.loss_utils import Regularizer
from src.utils.utils import load_model_from_file, robust_to_device


class LinearProbe:
    """Class for training a linear probe, with optional pre-trained, frozen model in front."""

    def __init__(
        self,
        reg_lambda: float,
        lr: float,
        epochs: int,
        device: str,
        seed: int,
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
    ) -> None:
        self.reg = Regularizer(regularization, reg_lambda)
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.model = None
        self.use_premodel = None
        self.logit_filter = logit_filter
        self.use_data_parallel = use_data_parallel
        self.filename = filename
        # TODO: figure out if we want to also have a separate logic to store the backbone, i.e., if it is fine-tuned
        self.force_train = force_train
        self.use_class_weights = use_class_weights
        self.min_learning_steps = min_learning_steps
        self.grad_norm_clip = grad_norm_clip

        if premodel is not None and not isinstance(premodel, (str, Path, torch.nn.Module, ThingsvisionModel)):
            raise ValueError(
                f"The passed premodel must be either a path to a model checkpoint"
                "or a torch.nn.Module or a ThingsvisionModel"
            )

        self.premodel = premodel
        self.freeze_premodel = freeze_premodel

    @staticmethod
    def assign_learning_rate(param_group: dict, new_lr: float) -> None:
        """Assign a new learning rate to a parameter group."""
        param_group["lr"] = new_lr

    @staticmethod
    def _warmup_lr(base_lr: float, warmup_length: float | int, step: int) -> float:
        """Warmup the learning rate."""
        return base_lr * (step + 1) / warmup_length

    def cosine_lr(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: float | list[float],
        warmup_length: int,
        steps: int,
        min_lr: float = 1e-6,
    ) -> Callable[[int], None]:
        """Cosine learning rate scheduler."""
        if not isinstance(base_lrs, list):
            base_lrs = [base_lrs for _ in optimizer.param_groups]

        if len(base_lrs) != len(optimizer.param_groups):
            raise ValueError(
                f"Length of base_lrs ({len(base_lrs)}) does not match number of parameter groups "
                f"({len(optimizer.param_groups)})"
            )
        min_lrs = [min_lr* lr / max(base_lrs) for lr in base_lrs]
        
        def _lr_adjuster(step: int) -> None:
            """Adjust the learning rate for the given step."""
            for param_group, base_lr, min_lr in zip(optimizer.param_groups, base_lrs, min_lrs, strict=True):
                if step < warmup_length:
                    lr = self._warmup_lr(base_lr, warmup_length, step)
                else:
                    e = step - warmup_length
                    es = steps - warmup_length
                    lr = 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr) + min_lr
                self.assign_learning_rate(param_group, lr)

        return _lr_adjuster

    def _init_new_model(
        self,
        input_shape: int | list[int],
        output_shape: int,
        premodel: torch.nn.Module | None = None,
    ) -> torch.nn.Module | torch.nn.DataParallel:
        """Initialize a new model."""
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        model = torch.nn.Linear(input_shape, output_shape)
        if premodel is not None:
            model = torch.nn.Sequential(premodel, model)
        model = model.to(self.device)
        if self.use_data_parallel and torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        return model

    def _get_input_output_shape(
        self, dataloader: DataLoader, premodel: torch.nn.Module | None = None
    ) -> tuple[int | list[int], int]:
        if hasattr(dataloader.dataset, "classes"):
            output_shape = len(dataloader.dataset.classes)
        elif hasattr(dataloader.dataset, "targets"):
            output_shape = dataloader.dataset.targets.max().item() + 1
        else:
            raise ValueError("Dataset does not have targets or classes attribute. Cannot proceed.")

        if premodel is None:
            if hasattr(dataloader.dataset, "feature_dims"):
                input_shape = dataloader.dataset.feature_dims
            else:
                raise ValueError(
                    "If no premodel is used than it is required to use a FeatureDataset, as the probe models expect prextracted features"
                )
        else:
            logger.info(f"Getting probe input shape by pushing a sample through the premodel.")
            for x, _ in dataloader:
                with torch.no_grad():
                    x = robust_to_device(x, self.device)
                    sample_output = premodel(x)
                input_shape = sample_output.shape[-1]
                break
        return input_shape, output_shape

    def _load_premodel(self) -> tuple[torch.nn.Module | None, bool]:
        """Load a premodel from a file and freeze its parameters if requested."""
        if self.premodel is None:
            return None, False

        if isinstance(self.premodel, str) or isinstance(self.premodel, Path):
            if Path(self.premodel).exists():
                premodel, use_premodel = (
                    load_model_from_file(self.premodel, self.device, use_data_parallel=False),
                    True,
                )
            else:
                raise ValueError(f"Premodel could not be found under {self.premodel=}, please provide a correct path")
        elif isinstance(self.premodel, (torch.nn.Module, ThingsvisionModel)):
            premodel, use_premodel = self.premodel, True
            if hasattr(self.premodel, "reset_to_pretrained"):
                premodel.reset_to_pretrained()
        else:
            raise ValueError(
                "Passed premodel not of type torch.nn.Module or ThingsvisionModel. Cannot proceed using the premodel."
            )
        if premodel is not None:
            if self.freeze_premodel:
                for param in premodel.parameters():
                    param.requires_grad = False
                logger.info("Premodel parameters frozen (requires_grad=False)")
            else:
                for param in premodel.parameters():
                    param.requires_grad = True
                logger.info("Premodel parameters unfrozen (requires_grad=True)")
        return premodel, use_premodel

    def _load_model(self, dataloader: DataLoader) -> bool:
        premodel, self.use_premodel = self._load_premodel()

        if self.filename is not None and Path(self.filename).exists() and not self.force_train:
            self.model = load_model_from_file(self.filename, self.device, use_data_parallel=False)

            if premodel is not None:
                self.model = torch.nn.Sequential(premodel, self.model)

            if self.use_data_parallel:
                self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
            return True

        input_shape, output_shape = self._get_input_output_shape(dataloader=dataloader, premodel=premodel)

        self.model = self._init_new_model(input_shape, output_shape, premodel=premodel)

        return False

    def _save_model(self) -> None:
        """Save the model. If the model is a sequential, we save the last layer.
        If the model is a DataParallel, we save the module.
        """
        curr_filename = Path(self.filename)
        if not curr_filename.parent.exists():
            curr_filename.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {curr_filename}")

        curr_model = self.model
        if isinstance(curr_model, torch.nn.DataParallel):
            curr_model = curr_model.module

        if self.use_premodel and isinstance(curr_model, torch.nn.Sequential) and self.freeze_premodel:
            curr_model = curr_model[-1]

        torch.save(curr_model, self.filename)

    def _get_criterion(self, dataloader: DataLoader) -> torch.nn.Module:
        """Get cross-entropy loss criterion potentially with class weights."""
        if self.use_class_weights and hasattr(dataloader.dataset, "targets"):
            targets = dataloader.dataset.targets
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            class_counts = torch.bincount(targets)
            total_count = class_counts.sum()
            class_weights = torch.where(class_counts == 0, 0, total_count / (len(class_counts) * class_counts))
            class_weights = class_weights.to(self.device)
            logger.info(f"Initializing cross-entropy loss WITH class weights.")
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            if self.use_class_weights:
                logger.warning(
                    "Wanted to use class weights, but dataset does not have targets attribute. "
                    "Using uniform class weights."
                )
            logger.info("Initializing cross-entropy loss WITHOUT class weights.")
            criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def set_epochs(self, epochs: int) -> None:
        """Set the number of epochs for the probe."""
        old_epochs = self.epochs
        self.epochs = epochs
        logger.info(f"Setting number of epochs from {old_epochs} to {self.epochs}.")

    def reinit_model(self, dataloader: DataLoader, params_to_set: dict[str, Any] | None = None) -> None:
        """Reinitialize the model with the given parameters."""
        logger.info(f"Reinitializing model model.")
        for name, param in params_to_set.items():
            if hasattr(self, name):
                setattr(self, name, param)
            elif hasattr(self.reg, name):
                setattr(self.reg, name, param)
            else:
                logger.warning(f"Parameter {name} not found in model and will therefore not be set. Continuing ...")
        self._load_model(dataloader)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer for the model."""
        if self.use_premodel and not self.freeze_premodel:
            # NOTE: cannot use pretrained end2end model to continue training
            back_bone_lr_factor = 0.001 #original 0.01 everywhere, however for fer we need 0.001
            return torch.optim.AdamW(
                [
                    {
                        "params": (p for p in self.model[0].parameters() if p.requires_grad),
                        "lr": self.lr * back_bone_lr_factor,
                        "weight_decay": 1e-6,
                    },
                    {
                        "params": (p for p in self.model[1].parameters() if p.requires_grad),
                        "lr": self.lr,
                        "weight_decay": self.reg.get_lambda(),
                    },
                ]
            ), [self.lr * back_bone_lr_factor, self.lr]
        else:
            return torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=self.lr,
                weight_decay=self.reg.get_lambda(),
            ), self.lr

    def train(
        self,
        dataloader: DataLoader,
        logger_every_n_epochs: int = 1,
    ) -> torch.nn.Module:
        """Train the model."""
        torch.manual_seed(self.seed)

        is_pretrained = self._load_model(dataloader)
        if is_pretrained:
            logger.info(f"Loaded pretrained model from {self.filename}")
            return self.model

        logger.info(f"Using {self.reg.reg_type} regularization with lambda: {self.reg.reg_lambda}")
        optimizer, base_lrs = self._get_optimizer()
        criterion = self._get_criterion(dataloader)

        logger.info(f"Optimizer has {len(optimizer.param_groups)} parameter group(s)")
        for idx, pg in enumerate(optimizer.param_groups):
            num_params = sum(p.numel() for p in pg["params"])
            logger.info(
                f"  Group {idx}: {num_params} params, lr={pg['lr']:.6e}, weight_decay={pg.get('weight_decay', 0):.6e}"
            )

        len_loader = len(dataloader)
        if len_loader * self.epochs < self.min_learning_steps:
            new_epochs = int(np.ceil(self.min_learning_steps / len_loader))
            logger.warning(
                f"Training results in only {(len_loader * self.epochs)=} steps. "
                f"We will increase the number of epochs to max({new_epochs}, 1) for better training."
            )
            self.epochs = max(new_epochs, 1)

        scheduler = self.cosine_lr(optimizer, base_lrs, 0.0, self.epochs * len_loader)

        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(dataloader):
                x = robust_to_device(x, self.device)
                y = y.to(self.device)
                step = i + epoch * len_loader
                scheduler(step)

                optimizer.zero_grad()

                pred = self.model(x)
                loss = criterion(pred, y)
                loss += self.reg.reg_loss(self.model)

                loss.backward()

                if self.grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)

                optimizer.step()

            if epoch % logger_every_n_epochs == 0:
                lr_str = ", ".join([f"LR_{i}: {pg['lr']:.5e}" for i, pg in enumerate(optimizer.param_groups)])
                logger.info(f"Train Epoch: {epoch + 1}/{self.epochs} \tLoss: {loss.item():.5f}\t{lr_str}")
        lr_str = ", ".join([f"LR_{i}: {pg['lr']:.5e}" for i, pg in enumerate(optimizer.param_groups)])
        logger.info(f"Train Epoch: {epoch + 1}/{self.epochs} \tLoss: {loss.item():.5f}\t{lr_str}")

        if self.filename is not None:
            try:
                self._save_model()
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                logger.error(f"Continuing training without saving model.")

        return self.model

    def infer(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference using the (trained) model on the provided dataloader."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        true, pred = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = robust_to_device(x, self.device)
                y = y.to(self.device)
                logits = self.model(x)
                if self.logit_filter is not None:
                    logits = logits @ self.logit_filter.T

                pred.append(logits.cpu())
                true.append(y.cpu())
        self.model.train()
        logits = torch.cat(pred)
        target = torch.cat(true)
        return logits, target
