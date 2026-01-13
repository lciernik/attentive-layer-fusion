import abc
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loader import get_rep2rep_feature_dl
from src.data.data_utils import SubsetWithTargets
from src.data.feature_combiner import get_feature_combiner_cls
from src.eval.metrics import (
    RepresentationMetrics,
    compute_metrics,
)
from src.models.mvae import SimpleMVAEModel, get_mvae_loss
from src.utils.loss_utils import Regularizer, get_loss
from src.utils.utils import check_feature_existence, load_model_from_file

from . import CombinedModelEvaluator, SingleModelEvaluator


class BaseRepTransfer(abc.ABC):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        lrs: List[float],
        epochs: int,
        seed: int,
        device: str,
        feature_dirs: Optional[List[str]],
        model_dirs: Optional[List[str]],
        results_dir: str,
        normalize: bool = True,
        val_proportion: float = 0,
        reg_lambda: float = 0.0,
        regularization: str = "weight_decay",
        force_train: bool = False,
        rep_loss: str = "mse",
        use_data_parallel: bool = False,
        eval_with_lin_probe: bool = True,
        fewshot_k: int = -1,
        rep_metrics_max_samples: int = 5000,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.rep_loss = rep_loss
        self.feature_dirs = feature_dirs

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA is not available, please run on a machine with a GPU.")
            raise RuntimeError("CUDA is not available, please run on a machine with a GPU.")

        logger.info(
            f"Running on device: {self.device}, cuda available: {torch.cuda.is_available()}, devices {torch.cuda.device_count()}"
        )
        self.normalize = normalize
        self.fewshot_k = fewshot_k

        # Get single model linear probe filenames
        assert model_dirs is not None, "Model directories must be provided for rep2rep task."
        self.single_linear_probe_fns = self._build_single_linear_probe_fns(model_dirs)

        # Create the transfer model path and the predictions directory
        self.transfer_model_path = os.path.join(results_dir, "model.pkl")
        self.linear_probe_path = results_dir if eval_with_lin_probe else None
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        self.val_proportion = val_proportion
        self.lrs = lrs
        self.reg_lambda = reg_lambda
        self.regularization = regularization
        self.force_train = force_train
        self.lr = None
        self.use_data_parallel = use_data_parallel
        self.rep_metrics = RepresentationMetrics(
            max_samples_similarity=rep_metrics_max_samples,  # TODO this is not accurate (but faster), better use 20k
            contrastive_temperature=1.0,
            device=self.device,
            criterion=self.rep_loss,
        )

    @staticmethod
    def _build_single_linear_probe_fns(model_dirs: List[str]) -> List[str]:
        # TODO Unfortunately, this model_dirs folder is not correct at the moment!
        # This accesses the old project directory
        model_dir = "/home/space/diverse_priors/models"
        model_slug = "no_fewshot/fewshot_epochs_20/regularization_weight_decay/batch_size_1024/seed_0"
        ds, model1 = model_dirs[0].split("models/")[-1].split("/loss")[0].split("/")
        ds, model2 = model_dirs[1].split("models/")[-1].split("/loss")[0].split("/")
        if "imagenet" in ds:
            # We do not have probes for ever imagenet subset but only for the overall dataset
            ds = "wds_imagenet1k"
        single_linear_probe_fns = [
            f"{model_dir}/{ds}/{model1}/{model_slug}/model.pkl",
            f"{model_dir}/{ds}/{model2}/{model_slug}/model.pkl",
        ]
        for fn in single_linear_probe_fns:
            if not os.path.exists(fn):
                logger.error(f"Single model linear probe not found: {fn}")
                raise FileNotFoundError(f"Single model linear probe not found: {fn}")
        return single_linear_probe_fns

    def require_feature_existence(self, check_train: bool = True):
        available_features = [
            check_feature_existence(feature_dir, check_train=check_train) for feature_dir in self.feature_dirs
        ]
        if not all(available_features):
            not_available_features = [
                feature_dir for feature_dir, available in zip(self.feature_dirs, available_features) if not available
            ]
            raise ValueError(f"Features are missing in {not_available_features}, please run single evaluator first!")

    @staticmethod
    def assign_learning_rate(param_group: dict, new_lr: float):
        param_group["lr"] = new_lr

    @staticmethod
    def _warmup_lr(base_lr: float, warmup_length: Union[float, int], step: int):
        return base_lr * (step + 1) / warmup_length

    def cosine_lr(self, optimizer, base_lrs, warmup_length, steps):
        if not isinstance(base_lrs, list):
            base_lrs = [base_lrs for _ in optimizer.param_groups]
        assert len(base_lrs) == len(optimizer.param_groups)

        def _lr_adjuster(step):
            for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
                if step < warmup_length:
                    lr = self._warmup_lr(base_lr, warmup_length, step)
                else:
                    e = step - warmup_length
                    es = steps - warmup_length
                    lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
                self.assign_learning_rate(param_group, lr)

        return _lr_adjuster

    def _create_train_val_loaders(self, train_loader: DataLoader) -> Tuple[DataLoader, DataLoader]:
        train_dataset = train_loader.dataset
        targets = np.array(train_dataset.targets)
        train_indices, val_indices = train_test_split(
            np.arange(targets.shape[0]),
            test_size=self.val_proportion,
            stratify=targets,
            random_state=self.seed,
        )
        tmp_train_dataset = SubsetWithTargets(train_dataset, indices=train_indices)
        tmp_val_dataset = SubsetWithTargets(train_dataset, indices=val_indices)

        tmp_train_loader = DataLoader(
            tmp_train_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
        )
        tmp_val_loader = DataLoader(
            tmp_val_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
        )
        return tmp_train_loader, tmp_val_loader

    def optimize_hyperparams(self, train_loader: DataLoader) -> None:
        """Optimizes the learning rate for the transfer model."""
        if self.val_proportion > 0:
            logger.info("\nTuning hyperparameters of transfer model.\n")
            # Split train set into train and validation
            tmp_train_loader, tmp_val_loader = self._create_train_val_loaders(train_loader)
            # Allow just different learning rates for now
            best_metric = float("inf")
            for lr in self.lrs:
                self.lr = lr
                logger.info(f"Evaluating learning rate: {lr}")
                val_metrics = self._evaluate(
                    tmp_train_loader,
                    tmp_val_loader,
                    evaluate_train=False,
                    evaluate_probes=False,
                )["test_metrics"]
                cur_metric = val_metrics["loss"]
                if cur_metric < best_metric:
                    best_metric = cur_metric
                    best_lr = lr
            self.lr = best_lr
            logger.info(f"Found best learning rate: {best_lr} with {self.rep_loss}: {best_metric}")
        else:
            if len(self.lrs) != 1:
                raise ValueError("Only one learning rate is supported without a validation set.")
            best_lr = self.lrs[0]

        self.lr = best_lr

    def _evaluate(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        filename: Optional[str] = None,
        evaluate_train: bool = True,
        evaluate_probes: bool = True,
    ) -> dict:
        train_loss = self.train(train_loader, filename=filename, force_train=self.force_train)

        metric_dict = {
            "reg_lambda": self.reg_lambda,
            "learning_rate": self.lr,
            "train_loss": train_loss,
        }

        if evaluate_probes:
            class_head_1 = load_model_from_file(
                self.single_linear_probe_fns[0],
                self.device,
                use_data_parallel=self.use_data_parallel,
            )
            class_head_2 = load_model_from_file(
                self.single_linear_probe_fns[1],
                self.device,
                use_data_parallel=self.use_data_parallel,
            )
        else:
            class_head_1, class_head_2 = None, None

        if evaluate_train:
            logger.info("Evaluating train set.")
            train_pred_dict = self.infer(train_loader, model_1=class_head_1, model_2=class_head_2)
            train_metrics = self.rep_metrics.compute_metrics(train_pred_dict["pred"], train_pred_dict["true"], split="train")
            metric_dict["train_metrics"] = train_metrics
            for k, v in train_pred_dict.items():
                if k in ["F1H1", "TF1H2", "F2H2"]:
                    metric_dict[f"train_{k}"] = compute_metrics(v, train_pred_dict["true_class"], split="train")

        logger.info("Evaluating test set.")
        test_pred_dict = self.infer(test_loader, model_1=class_head_1, model_2=class_head_2)
        test_metrics = self.rep_metrics.compute_metrics(test_pred_dict["pred"], test_pred_dict["true"], split="test")
        metric_dict["test_metrics"] = test_metrics

        for k, v in test_pred_dict.items():
            if k in ["F1H1", "TF1H2", "F2H2"]:
                metric_dict[f"test_{k}"] = compute_metrics(v, test_pred_dict["true_class"], split="test")
        # Save the Tuned Hyperparameters
        metric_dict["LR"] = self.lr
        metric_dict["reg_lambda"] = self.reg_lambda

        if self.linear_probe_path is not None:
            logger.info("Evaluating linear probe on transformed features.")
            logger.info(f"> Using feature dirs: {self.feature_dirs[:1]}")
            evaluator = SingleModelEvaluator(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                lrs=self.lrs,
                epochs=self.epochs,
                seed=self.seed,
                device=self.device,
                fewshot_k=self.fewshot_k,
                feature_dirs=self.feature_dirs[:1],
                model_dirs=[self.linear_probe_path],
                results_dir=self.linear_probe_path,
                val_proportion=self.val_proportion,
                reg_lambda=self.reg_lambda,
                regularization=self.regularization,
                force_train=self.force_train,
                model_fn="probe.pkl",
                premodel_filename=self.transfer_model_path,
            )
            probe_metrics = evaluator.evaluate()
            metric_dict["linear_probe_eval"] = probe_metrics

        return metric_dict

    def save_model(self, filename: str) -> None:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.info(f"Saving model to {filename}")
        torch.save(self.model, filename)

    def load_model(self, filename: str) -> float:
        logger.info(f"Loading pretrained model from {filename=}")
        self.model = load_model_from_file(filename, self.device, use_data_parallel=self.use_data_parallel)
        logger.info("Using pretrained model - returning nan for training loss")
        return np.nan

    def evaluate(self) -> dict:
        probe_exists = os.path.exists(self.transfer_model_path) and not self.force_train

        self.require_feature_existence(check_train=not probe_exists)

        feature_train_loader, feature_test_loader = get_rep2rep_feature_dl(
            feature_dirs=self.feature_dirs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            fewshot_k=self.fewshot_k,
            load_train=True,
            normalize=self.normalize
        )

        if probe_exists:
            self.reg_lambda = None
        else:
            self.optimize_hyperparams(feature_train_loader)

        return self._evaluate(
            train_loader=feature_train_loader,
            test_loader=feature_test_loader,
            filename=self.transfer_model_path,
            evaluate_train=True,
        )

    @abc.abstractmethod
    def infer(
        self,
        dataloader: DataLoader,
        model_1: Optional[torch.nn.Module] = None,
        model_2: Optional[torch.nn.Module] = None,
    ) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def train(
        self,
        dataloader: DataLoader,
        filename: Optional[str] = None,
        force_train: bool = False,
        logger_every_n_epochs: int = 10,
    ) -> float:
        raise NotImplementedError("Subclasses must implement this method")


class LinearRepTransfer(BaseRepTransfer):
    def _init_new_model(self, input_shape: int, output_shape: int) -> torch.nn.Module:
        model = torch.nn.Linear(input_shape, output_shape)
        model = model.to(self.device)
        if self.use_data_parallel and torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=[x for x in range(torch.cuda.device_count())])
        return model

    def train(
        self,
        dataloader: DataLoader,
        filename: Optional[str] = None,
        force_train: bool = False,
        logger_every_n_epochs: int = 10,
    ) -> float:
        # We reset the seed to ensure that the model is initialized with the same weights every time
        torch.manual_seed(self.seed)

        if filename is not None and os.path.exists(filename) and not force_train:
            return self.load_model(filename)

        input_shape, output_shape = (
            dataloader.dataset[0][0].shape[0],
            dataloader.dataset[0][1].shape[0],
        )
        self.model = self._init_new_model(input_shape, output_shape)

        self.reg = Regularizer(self.regularization, self.reg_lambda)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.reg.get_lambda(),
        )
        criterion = get_loss(self.rep_loss, batch_size=self.batch_size, device=self.device)
        len_loader = len(dataloader)
        scheduler = self.cosine_lr(optimizer, self.lr, 0.0, self.epochs * len_loader)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for i, (x, y, _) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                step = i + epoch * len_loader
                scheduler(step)

                optimizer.zero_grad()

                pred = self.model(x)
                if self.normalize:
                    pred = torch.nn.functional.normalize(pred, dim=1)

                loss = criterion(pred, y)
                loss += self.reg.reg_loss(self.model)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len_loader
            if epoch % logger_every_n_epochs == 0:
                logger.info(
                    f"Train Epoch: {epoch} \tLoss: {epoch_loss:.6f}\tLR {optimizer.param_groups[0]['lr']:.5f}",
                )

        if filename is not None:
            self.save_model(filename)
        return epoch_loss

    def infer(
        self,
        dataloader: DataLoader,
        model_1: Optional[torch.nn.Module] = None,
        model_2: Optional[torch.nn.Module] = None,
    ) -> dict:
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        pred_dict = {"true": [], "pred": []}
        if model_1 is not None and model_2 is not None:
            pred_dict.update({"true_class": [], "F1H1": [], "TF1H2": [], "F2H2": []})
        with torch.no_grad():
            for x, y, c in tqdm(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Predict Transferred Features
                pred_features = self.model(x)
                pred_dict["true"].append(y.cpu())
                pred_dict["pred"].append(pred_features.cpu())
                if model_1 is not None and model_2 is not None:
                    pred_dict["true_class"].append(c.cpu())
                    pred_dict["F1H1"].append(model_1(x).cpu())
                    pred_dict["TF1H2"].append(model_2(pred_features).cpu())
                    pred_dict["F2H2"].append(model_2(y).cpu())

        logger.info(
            f"Inference done for {len(dataloader)} batches {'with' if model_1 is not None else 'without'} classification probes."
        )
        output = {k: torch.cat(v) for k, v in pred_dict.items()}
        return output


class MVAERepTransfer(BaseRepTransfer):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        lrs: List[float],
        epochs: int,
        seed: int,
        device: str,
        feature_dirs: Optional[List[str]],
        model_dirs: Optional[List[str]],
        results_dir: Optional[str],
        normalize: bool = True,
        val_proportion: float = 0,
        reg_lambda: float = 0.0,
        regularization: str = "weight_decay",
        force_train: bool = False,
        rep_loss: str = "mse",
        use_data_parallel: bool = False,
        eval_with_lin_probe: bool = True,
        fewshot_k: int = -1,
    ) -> None:
        super().__init__(
            batch_size,
            num_workers,
            lrs,
            epochs,
            seed,
            device,
            feature_dirs,
            model_dirs,
            results_dir,
            normalize,
            val_proportion,
            reg_lambda,
            regularization,
            force_train,
            rep_loss,
            use_data_parallel,
            eval_with_lin_probe,
            fewshot_k,
        )
        self.mvae_loss_type, self.combine_z, loss_loose, self.used_distribution = rep_loss.split("_")
        self.loss_loose = loss_loose == "loose"

    def train(
        self,
        dataloader: DataLoader,
        filename: Optional[str] = None,
        force_train: bool = False,
        logger_every_n_epochs: int = 10,
    ) -> float:
        # We reset the seed to ensure that the model is initialized with the same weights every time
        torch.manual_seed(self.seed)

        if filename is not None and os.path.exists(filename) and not force_train:
            return self.load_model(filename)

        self.reg = Regularizer(self.regularization, self.reg_lambda)
        x_shape, y_shape = (
            dataloader.dataset[0][0].shape[0],
            dataloader.dataset[0][1].shape[0],
        )

        self.model = SimpleMVAEModel(
            x_shape,
            y_shape,
            self.device,
            self.combine_z,
            self.used_distribution,
            linear="NL" not in self.mvae_loss_type,
            normalize="normalize" in self.mvae_loss_type,
            individual_variance="IndVariance" in self.mvae_loss_type,
        )

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.reg.get_lambda(),
        )

        criterion = get_mvae_loss(
            method=self.combine_z,
            loose=self.loss_loose,
            distribution=self.used_distribution,
            loss_modification=self.mvae_loss_type,
        )

        len_loader = len(dataloader)
        scheduler = self.cosine_lr(optimizer, self.lr, 0.0, self.epochs * len_loader)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for i, (x, y, _) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                step = i + epoch * len_loader
                scheduler(step)

                optimizer.zero_grad()
                loss = criterion(x, y, self.model)

                loss += self.reg.reg_loss(self.model)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len_loader
            if epoch % logger_every_n_epochs == 0:
                logger.info(
                    f"Train Epoch: {epoch} \tLoss: {epoch_loss:.6f}\tLR {optimizer.param_groups[0]['lr']:.5f}",
                )

        if filename is not None:
            self.save_model(filename)
        return epoch_loss

    def infer(
        self,
        dataloader: DataLoader,
        model_1: Optional[torch.nn.Module] = None,
        model_2: Optional[torch.nn.Module] = None,
        max_batches: int = 200,
    ) -> dict:
        """Make inference of the transfer model.
        Max batches is used to limit the number of batches to evaluate, e.g.,
        ImageNet has 1.28M images, and we cannot evaluate multiple views of the same image.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        pred_dict = {
            "true_x": [],
            "rec_x|x": [],
            "rec_x|y": [],
            "rec_x|x,y": [],
            "true_y": [],
            "rec_y|x": [],
            "rec_y|y": [],
            "rec_y|x,y": [],
            "z_joint": [],
            "z|x": [],
            "z|y": [],
            "z_scale": [],
        }
        if model_1 is not None and model_2 is not None:
            pred_dict.update({"true_class": [], "X->C": [], "Y->C": [], "RX->C": [], "RY->C": []})
        with torch.no_grad():
            for x, y, c in tqdm(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Predict Transferred Features
                q1, q2 = self.model.encode(x, y)
                rec_x_x, rec_y_x = self.model.decode(q1.mean)
                rec_x_y, rec_y_y = self.model.decode(q2.mean)
                z_joint = self.model.combine_representations(q1, q2)
                rec_x_joint, rec_y_joint = self.model.decode(z_joint.mean)
                pred_dict["true_x"].append(x.cpu())
                pred_dict["rec_x|x"].append(rec_x_x.cpu())
                pred_dict["rec_x|y"].append(rec_x_y.cpu())
                pred_dict["rec_x|x,y"].append(rec_x_joint.cpu())
                pred_dict["true_y"].append(y.cpu())
                pred_dict["rec_y|x"].append(rec_y_x.cpu())
                pred_dict["rec_y|y"].append(rec_y_y.cpu())
                pred_dict["rec_y|x,y"].append(rec_y_joint.cpu())
                pred_dict["z_joint"].append(z_joint.mean.cpu())
                pred_dict["z|x"].append(q1.mean.cpu())
                pred_dict["z|y"].append(q2.mean.cpu())
                pred_dict["z_scale"].append(z_joint.scale.cpu())
                if model_1 is not None and model_2 is not None:
                    pred_dict["true_class"].append(c.cpu())
                    pred_dict["X->C"].append(model_1(x).cpu())
                    pred_dict["Y->C"].append(model_2(y).cpu())
                    pred_dict["RX->C"].append(model_1(rec_x_joint).cpu())
                    pred_dict["RY->C"].append(model_2(rec_y_joint).cpu())
                if len(pred_dict["true_x"]) >= max_batches:
                    break
        logger.info(
            f"Inference done for {len(pred_dict['true_x'])} batches {'with' if model_1 is not None else 'without'} classification probes.",
        )
        output = {k: torch.cat(v) for k, v in pred_dict.items()}
        return output

    def _evaluate(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        filename: Optional[str] = None,
        evaluate_train: bool = True,
        evaluate_probes: bool = True,
    ) -> dict:
        train_loss = self.train(train_loader, filename=filename, force_train=self.force_train)

        metric_dict = {
            "reg_lambda": self.reg_lambda,
            "learning_rate": self.lr,
            "train_loss": train_loss,
        }
        if not evaluate_train:
            # This is used only during hyper parameter training, so we skip the more expensive evaluation protocols
            test_loss = self.infer_only_loss(test_loader)
            metric_dict["test_metrics"] = {"loss": test_loss}
            return metric_dict

        if evaluate_probes:
            class_head_1 = load_model_from_file(
                self.single_linear_probe_fns[0],
                self.device,
                use_data_parallel=self.use_data_parallel,
            )
            class_head_2 = load_model_from_file(
                self.single_linear_probe_fns[1],
                self.device,
                use_data_parallel=self.use_data_parallel,
            )
        else:
            class_head_1, class_head_2 = None, None

        train_pred_dict = self.infer(train_loader, model_1=class_head_1, model_2=class_head_2)
        metric_dict = self._agregate_metrics(train_pred_dict, metric_dict, mode="train")

        test_pred_dict = self.infer(test_loader, model_1=class_head_1, model_2=class_head_2)
        metric_dict = self._agregate_metrics(test_pred_dict, metric_dict, mode="test")

        # Save the Tuned Hyperparameters
        metric_dict["LR"] = self.lr
        metric_dict["reg_lambda"] = self.reg_lambda

        if self.linear_probe_path is not None:
            logger.info("Evaluating linear probe on transformed features.")
            logger.info(f"> Using feature dirs: {self.feature_dirs}")
            feature_combiner_cls = get_feature_combiner_cls("tuple")
            evaluator = CombinedModelEvaluator(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                lrs=self.lrs,
                epochs=self.epochs,
                seed=self.seed,
                device=self.device,
                fewshot_k=self.fewshot_k,
                feature_dirs=self.feature_dirs,
                model_dirs=[self.linear_probe_path],
                results_dir=self.linear_probe_path,
                val_proportion=self.val_proportion,
                reg_lambda=self.reg_lambda,
                regularization=self.regularization,
                force_train=self.force_train,
                feature_combiner_cls=feature_combiner_cls,
                model_fn="Combinedprobe.pkl",
                premodel_filename=self.transfer_model_path,
            )
            probe_metrics = evaluator.evaluate()
            metric_dict["linear_probe_eval"] = probe_metrics

        return metric_dict

    def infer_only_loss(self, dataloader: DataLoader) -> float:
        epoch_loss = 0
        criterion = get_mvae_loss(
            method=self.combine_z,
            loose=self.loss_loose,
            distribution=self.used_distribution,
            loss_modification=self.mvae_loss_type,
        )

        with torch.no_grad():
            for i, (x, y, _) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                loss = criterion(x, y, self.model)
                epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    def _agregate_metrics(self, pred_dict: dict, metric_dict: dict, mode: str = "test") -> dict:
        eval_combinations = [
            ("x|x", "rec_x|x", "true_x", True),
            ("x|y", "rec_x|y", "true_x", True),
            ("x|x,y", "rec_x|x,y", "true_x", True),
            ("y|x", "rec_y|x", "true_y", True),
            ("y|y", "rec_y|y", "true_y", True),
            ("y|x,y", "rec_y|x,y", "true_y", True),
            ("z|x,y<->x", "z_joint", "true_x", False),
            ("z|x,y<->y", "z_joint", "true_y", False),
            ("z|x<->z|y", "z|x", "z|y", True),
            ("x<->y", "true_x", "true_y", False),
        ]
        for name, a, b, same in eval_combinations:
            logger.info(f"Computing metrics for {name} on the {mode} set.")
            tmp_metric = self.rep_metrics.compute_metrics(pred_dict[a], pred_dict[b], same_dim=same)
            metric_dict[f"{mode}_{name}"] = tmp_metric

        for k, v in pred_dict.items():
            if k in ["X->C", "Y->C", "RX->C", "RY->C"]:
                metric_dict[f"{mode}_{k}"] = compute_metrics(v, pred_dict["true_class"])

        return metric_dict
