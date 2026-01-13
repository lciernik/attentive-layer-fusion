import pickle
import time
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.data.data_utils import create_train_val_loaders
from src.eval.metrics import compute_metrics
from src.tasks.attentive_probe import AttentiveProbe
from src.tasks.hyperparameter_tuner import HyperparameterTuner
from src.tasks.linear_probe import LinearProbe


class BaseEvaluator:
    """Base class for all representation evaluators."""

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        lrs: list[float],
        probe_args: dict,
        seed: int,
        fewshot_k: int,
        model_dirs: list[str] | None,
        results_dir: str | None,
        normalize: bool = True,
        val_proportion: float = 0,
        force_train: bool = False,
        model_fn: str = "model.pkl",
        reg_lambda_bounds: tuple[int, int] = (-6, 0),

    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.device = probe_args.get("device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, please run on a machine with a GPU.")

        self.normalize = normalize
        self.fewshot_k = fewshot_k
        if model_dirs is not None:
            self.probe_fns = [str(Path(model_dir) / model_fn) for model_dir in model_dirs]
        else:
            self.probe_fns = None
        self.results_dir = results_dir
        self.val_proportion = val_proportion
        self.lrs = lrs
        self.force_train = force_train
        self.lr = None
        self.reg_lambda = probe_args.get("reg_lambda", 0.0)
        self.min_exp_lambda, self.max_exp_lambda = reg_lambda_bounds
        # Store the logit filter for the final trained probe
        self.logit_filter = probe_args.get("logit_filter")
        if "logit_filter" in probe_args:
            probe_args.pop("logit_filter")
        self.probe_type = probe_args.get("probe_type", "linear")
        if "probe_type" in probe_args:
            probe_args.pop("probe_type")

        self.probe = self._init_probe(
            probe_type=self.probe_type,
            lr=lrs[0] or 0.01,
            logit_filter=None,
            **probe_args,
        )

        self.hp_tuner = HyperparameterTuner(
            probe=self.probe,
            lrs=lrs,
        )

    def _init_probe(
        self,
        probe_type: str,
        **kwargs,
    ) -> LinearProbe | AttentiveProbe:
        common_params = {
            "reg_lambda": kwargs.get("reg_lambda", 0.0),
            "lr": kwargs.get("lr", 0.01),
            "epochs": kwargs.get("epochs", 10),
            "device": self.device,
            "seed": self.seed,
            "logit_filter": kwargs.get("logit_filter"),
            "regularization": kwargs.get("regularization", "weight_decay"),
            "use_data_parallel": kwargs.get("use_data_parallel", False),
            "filename": kwargs.get("filename"),
            "force_train": kwargs.get("force_train", False),
            "premodel": kwargs.get("premodel"),
            "freeze_premodel": kwargs.get("freeze_premodel", True),
            "use_class_weights": kwargs.get("use_class_weights", True),
            "grad_norm_clip": kwargs.get("grad_norm_clip", None),
        }

        if probe_type == "linear":
            return LinearProbe(
                **common_params,
            )
        elif probe_type == "attentive":
            return AttentiveProbe(
                **common_params,
                dim=kwargs["dim"],
                num_heads=kwargs.get("num_heads", 8),
                dimension_alignment=kwargs.get("dimension_alignment", "zero_padding"),
                always_project=kwargs.get("always_project", False),
                attention_dropout=kwargs.get("attention_dropout", (0.0, 0.0)),
            )
        else:
            raise ValueError(f"Unknown probe type: {probe_type}. Available options are: linear, attentive.")

    def optimize_hyperparams(
        self,
        train_loader: DataLoader,
        vals_between_init: int = 0,
        train_dataset_config: dict | None = None,
    ) -> float:
        """Optimize the hyperparameters of the selected probe (linear or attentive)."""
        if self.val_proportion > 0:
            logger.info("\nTuning hyperparameters of probe.\n")
            # Split train set into train and validation
            tmp_train_loader, tmp_val_loader = create_train_val_loaders(
                train_loader,
                train_dataset_config=train_dataset_config,
                val_proportion=self.val_proportion,
                seed=self.seed,
            )
            best_lr, best_wd, max_acc = self.hp_tuner.tune(
                tmp_train_loader, tmp_val_loader, self.min_exp_lambda, self.max_exp_lambda, vals_between_init
            )
        else:
            if len(self.lrs) != 1:
                raise ValueError("Only one learning rate is supported without a validation set.")
            logger.info(f"No validation proportion set, using predefined regularization lambda and learning rate.")
            best_lr = self.lrs[0]
            best_wd = self.reg_lambda
            max_acc = None

        self.lr = best_lr
        self.reg_lambda = best_wd
        logger.info(
            f"--- Best learning rate: {self.lr}, best regularization lambda: {self.reg_lambda}, max accuracy: {max_acc} ---"
        )
        return max_acc

    def load_test_set_predictions(self, probe_directory: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the test set predictions from the probe directory."""
        predictions_path = Path(probe_directory) / "predictions.pkl"
        with predictions_path.open("rb") as f:
            predictions = pickle.load(f)  # noqa: S301
            logits = predictions["logits"]
            target = predictions["target"]
            logger.info(f"Loaded test predictions from {predictions_path}.")
        return logits, target

    def store_test_set_predictions(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Method to store the test set predictions."""
        results_dir = Path(self.results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Create path to store predictions: {self.results_dir}")
        with (results_dir / "predictions.pkl").open("wb") as f:
            pickle.dump({"logits": logits, "pred": logits.argmax(dim=1), "target": target}, f)
        logger.info(f"Stored test predictions in {results_dir / 'predictions.pkl'}.")

    def _evaluate(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        filename: str | None = None,
        evaluate_train: bool = True,
    ) -> dict:
        metric_dict = {
            "reg_lambda": self.reg_lambda,
            "learning_rate": self.lr,
        }

        self.probe.reinit_model(
            train_loader,
            params_to_set={
                "reg_lambda": self.reg_lambda,
                "lr": self.lr,
                "logit_filter": self.logit_filter,
                "force_train": self.force_train,
                "filename": filename,
            },
        )
        st = time.time()
        self.probe.train(train_loader)
        et = time.time()

        metric_dict["training_epochs"] = self.probe.epochs

        if evaluate_train:
            metric_dict["training_time"] = et - st
            # inference time on training data
            st = time.time()
            train_logits, train_targets = self.probe.infer(train_loader)
            et = time.time()

            # compute metrics on training data
            train_metrics = compute_metrics(train_logits, train_targets, split="train")
            metric_dict["train_metrics"] = train_metrics
            metric_dict["train_data_inference_time"] = et - st

        # inference time on test data
        st = time.time()
        test_logits, test_targets = self.probe.infer(test_loader)
        et = time.time()
        metric_dict["test_data_inference_time"] = et - st

        # compute metrics on test data
        test_metrics = compute_metrics(test_logits, test_targets, split="test")
        metric_dict["test_metrics"] = test_metrics

        if train_loader is not None and self.batch_size != train_loader.batch_size:
            metric_dict["train_batch_size"] = train_loader.batch_size

        self.store_test_set_predictions(test_logits, test_targets)

        return metric_dict

    def evaluate(self) -> dict:
        """Evaluate the representations of the model."""
        raise NotImplementedError("Subclasses must implement this method")
