import time
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.data.data_loader import get_combined_feature_dl, get_feature_dl
from src.data.feature_combiner import (
    ConcatFeatureCombiner,
    PCAConcatFeatureCombiner,
    TupleFeatureCombiner,
)
from src.eval.metrics import compute_metrics
from src.models import ThingsvisionModel
from src.tasks.base_evaluator import BaseEvaluator
from src.utils.utils import check_feature_existence, check_single_instance


class SingleModelEvaluator(BaseEvaluator):
    """Evaluate representations of a single model."""

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        lrs: list[float],
        probe_args: dict,
        seed: int,
        fewshot_k: int,
        feature_dirs: list[str] | None,
        model_dirs: list[str] | None,
        results_dir: str | None,
        model: ThingsvisionModel | None = None,
        train_dataloader: DataLoader | None = None,
        eval_dataloader: DataLoader | None = None,
        normalize: bool = True,
        val_proportion: float = 0,
        force_train: bool = False,
        model_fn: str = "model.pkl",
        jitter_p: float = 0.5,
        reg_lambda_bounds: tuple[int, int] = (-6, 0),
    ) -> None:
        super().__init__(
            batch_size,
            num_workers,
            lrs,
            probe_args,
            seed,
            fewshot_k,
            model_dirs,
            results_dir,
            normalize,
            val_proportion,
            force_train,
            model_fn,
            reg_lambda_bounds,
        )

        self.feature_dir = check_single_instance(feature_dirs, "feature directory")
        self.probe_fn = check_single_instance(self.probe_fns, "probe filename")

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.jitter_p = jitter_p

    def _check_model_and_loaders_existence(self) -> bool:
        """Check if model and dataloaders are provided.
        If a model is provided at least one dataloader should be provided.
        """
        return (self.model is not None) and ((self.train_dataloader is not None) or (self.eval_dataloader is not None))

    def _extract_all_features(self, extract_train: bool = True, extract_test: bool = True) -> None:
        """Extract all features for the model."""
        if not self._check_model_and_loaders_existence():
            raise ValueError("Model and dataloaders are not provided. Cannot extract features.")
        if extract_test:
            logger.info("Extracting FEATURES for TEST set...")
            self.model.extract_features(self.eval_dataloader, self.feature_dir, split="_test")
            logger.info("Extracting TARGETS for TEST set...")
            self.model.extract_targets(self.eval_dataloader, self.feature_dir, split="_test")
        if extract_train:
            logger.info("Extracting FEATURES for TRAIN set...")
            self.model.extract_features(self.train_dataloader, self.feature_dir, split="_train")
            logger.info("Extracting TARGETS for TRAIN set...")
            self.model.extract_targets(self.train_dataloader, self.feature_dir, split="_train")

    def ensure_feature_availability(self, check_train: bool = True) -> None:
        """Ensures that the features and targets are available for the passed model.
        If the features are not available, it will extract them.
        """
        model_available = self._check_model_and_loaders_existence()
        check_train = check_train and self.train_dataloader is not None
        feat_available = check_feature_existence(
            self.feature_dir,
            module_names=self.model._module_names if model_available else None,
            check_train=check_train,
        )
        if (not feat_available) and model_available:
            logger.info(
                f"Features are not available yet. Extracting features for the model "
                f"{'(only for test) ' if not check_train else ''}..."
            )
            self._extract_all_features(extract_train=check_train)

        elif (not feat_available) and (not model_available):
            raise ValueError(
                f"Features are missing (in {self.feature_dir})\nand no model and dataloaders are provided."
                f"Please run feature extraction first."
            )
        else:
            logger.info(f"Features are already available in {self.feature_dir}.")

    def evaluate(self) -> dict:
        """Evaluate the model's representations.

        The method proceeds as follows:
        1. Checks if a pretrained probe file exists.
        2. Ensures that required features are available, extracting them if necessary.
        3. If no pretrained probe is found, optimizes probe hyperparameters using the training set.
        4. Evaluates the model's representations using the (possibly newly trained) probe.
        """
        probe_exists = Path(self.probe_fn).exists() and not self.force_train

        self.ensure_feature_availability(check_train=not probe_exists)

        feature_train_loader, feature_test_loader = get_feature_dl(
            feature_dir=self.feature_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            fewshot_k=self.fewshot_k,
            load_train=not probe_exists,
            normalize=self.normalize,
            jitter_p=self.jitter_p,
        )
        if probe_exists:
            logger.info(f"Found existing pretrained probe at {self.probe_fn=}")
            self.reg_lambda = None
            hopt_time_s = None
            best_val_bal_acc1 = None
        else:
            logger.info(f"Could not find pretrained probe at {self.probe_fn=}")
            logger.info("Optimizing hyperparameters of linear probe ...")
            st = time.time()
            best_val_bal_acc1 = self.optimize_hyperparams(feature_train_loader)
            et = time.time()
            hopt_time_s = et - st

        metric_dict = self._evaluate(
            train_loader=feature_train_loader,
            test_loader=feature_test_loader,
            filename=self.probe_fn,
            evaluate_train=not probe_exists,
        )
        metric_dict["hopt_time_s"] = hopt_time_s
        metric_dict["best_val_bal_acc1"] = best_val_bal_acc1
        return metric_dict


class CombinedModelEvaluator(BaseEvaluator):
    """Evaluate the representations of multiple models that will be combined."""

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        lrs: list[float],
        probe_args: dict,
        seed: int,
        fewshot_k: int,
        feature_dirs: list[str] | None,
        model_dirs: list[str] | None,
        results_dir: str | None,
        feature_combiner_cls: ConcatFeatureCombiner | PCAConcatFeatureCombiner | TupleFeatureCombiner,
        normalize: bool = True,
        val_proportion: float = 0,
        force_train: bool = False,
        model_fn: str = "model.pkl",
        jitter_p: float = 0.5,
        reg_lambda_bounds: tuple[int, int] = (-6, 0),
    ) -> None:
        super().__init__(
            batch_size,
            num_workers,
            lrs,
            probe_args,
            seed,
            fewshot_k,
            model_dirs,
            results_dir,
            normalize,
            val_proportion,
            force_train,
            model_fn,
            reg_lambda_bounds,
        )
        self.feature_dirs = feature_dirs
        self.feature_combiner_cls = feature_combiner_cls
        self.probe_fn = check_single_instance(self.probe_fns, "probe filename")

        self.jitter_p = jitter_p

    def require_feature_existence(self, check_train: bool = True) -> None:
        """Check if the features are available for the passed model.
        The combined model evaluator requires that all features are available for all models.
        If the features are not available, it will raise an error.
        """
        available_features = [
            check_feature_existence(feature_dir, check_train=check_train) for feature_dir in self.feature_dirs
        ]
        if not all(available_features):
            not_available_features = [
                feature_dir
                for feature_dir, available in zip(self.feature_dirs, available_features, strict=False)
                if not available
            ]
            raise ValueError(f"Features are missing in {not_available_features}, please run single evaluator first!")

    def evaluate(self) -> dict:
        """Evaluate the representations of the combined model.
        The method proceeds as follows:
        1. Checks if a pretrained probe file exists.
        2. Ensures that required features are available for all models.
        3. If no pretrained probe is found, optimizes probe hyperparameters using the training set.
        4. Evaluates the model's representations using the (possibly newly trained) probe.
        """
        probe_exists = Path(self.probe_fn).exists() and not self.force_train

        self.require_feature_existence(check_train=not probe_exists)

        feature_train_loader, feature_test_loader = get_combined_feature_dl(
            feature_dirs=self.feature_dirs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            fewshot_k=self.fewshot_k,
            feature_combiner_cls=self.feature_combiner_cls,
            jitter_p=self.jitter_p,
            normalize=self.normalize,
            load_train=not probe_exists,
        )

        if probe_exists:
            self.reg_lambda = None
            hopt_time_s = None
            best_val_bal_acc1 = None
        else:
            st = time.time()
            best_val_bal_acc1 = self.optimize_hyperparams(feature_train_loader)
            et = time.time()
            hopt_time_s = et - st

        metric_dict = self._evaluate(
            train_loader=feature_train_loader,
            test_loader=feature_test_loader,
            filename=self.probe_fn,
            evaluate_train=not probe_exists,
        )
        metric_dict["hopt_time_s"] = hopt_time_s
        metric_dict["best_val_bal_acc1"] = best_val_bal_acc1
        return metric_dict


class EnsembleModelEvaluator(BaseEvaluator):
    """Evaluate the representations of an ensemble of models."""

    # TODO: This class does not work as expected as the BaseEvaluator # noqa: TD003
    # has been refactored.
    # TODO: We need to update this class to work with the new BaseEvaluator. # noqa: TD003
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        lrs: list[float],
        probe_args: dict,
        seed: int,
        fewshot_k: int,
        model_ids: list[str],
        feature_dirs: list[str] | None,
        model_dirs: list[str] | None,
        results_dir: str | None,
        single_prediction_dirs: list[str] | None,
        normalize: bool = True,
        val_proportion: float = 0,
        force_train: bool = False,
        model_fn: str = "model.pkl",
    ) -> None:
        super().__init__(
            batch_size,
            num_workers,
            lrs,
            probe_args,
            seed,
            fewshot_k,
            model_dirs,
            results_dir,
            normalize,
            val_proportion,
            force_train,
            model_fn,
        )
        self.model_ids = model_ids
        self.feature_dirs = feature_dirs
        self.single_prediction_dirs = single_prediction_dirs
        if not len(model_ids) == len(feature_dirs) == len(single_prediction_dirs):
            raise ValueError(
                "Number of models, feature, single model prediction, and  linear probe directories must be the same."
            )

    def check_equal_targets(self, model_targets: dict[str, torch.Tensor]) -> None:
        """Check if the targets are the same across models.
        Method will raise an error if the targets are not the same.
        """
        if not all(
            torch.equal(model_targets[model_id], model_targets[self.model_ids[0]]) for model_id in self.model_ids
        ):
            raise ValueError("Targets are not the same across models.")

    @staticmethod
    def ensemble_logits(model_logits: dict[str, torch.Tensor], mode: str = "post_softmax") -> torch.Tensor:
        """Ensemble the logits of the models."""
        if mode == "post_softmax":
            # Softmax does not work for float16
            probs = torch.stack(
                [torch.nn.functional.softmax(logits.float(), dim=1) for logits in model_logits.values()],
                dim=0,
            )
            logits = torch.mean(probs, dim=0)
        elif mode == "pre_softmax":
            logits = torch.mean(torch.stack(list(model_logits.values()), dim=0), dim=1)
        else:
            raise ValueError(f"Unknown mode {mode}")
        return logits

    def evaluate(self) -> dict:
        """Evaluate the representations of the ensemble of models.
        The method proceeds as follows:
        1. Loads the predictions for each model.
        2. Checks if the targets are the same across models.
        3. Ensembles the logits of the models.
        4. Stores the predictions and computes the metrics.
        """
        model_logits = {}
        model_targets = {}
        for model_id, model_pred_dir in zip(self.model_ids, self.single_prediction_dirs, strict=True):
            # Try to load predictions directly for maximum speed
            pred_fn = Path(model_pred_dir) / "predictions.pkl"
            if pred_fn.is_file():
                logits, target = self.load_test_set_predictions(model_pred_dir)
                model_logits[model_id] = logits
                model_targets[model_id] = target
            else:
                raise ValueError(
                    f"Predictions for model {model_id} are missing, please run single evaluator for {model_id} first!"
                )

        self.check_equal_targets(model_targets)

        logits = self.ensemble_logits(model_logits)
        self.store_test_set_predictions(logits, model_targets[self.model_ids[0]])
        metric_dict = compute_metrics(logits, model_targets[self.model_ids[0]], split="test")
        metric_dict = {"test_metrics": metric_dict}
        return metric_dict
