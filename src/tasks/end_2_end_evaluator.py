import time
from pathlib import Path

import torch
import webdataset as wds
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.models import ThingsvisionModel
from src.tasks.base_evaluator import BaseEvaluator
from src.utils.utils import check_single_instance


class End2endModelEvaluator(BaseEvaluator):
    """In comparison to the SingleModelEvalutor, the End2endModelEvaluator
    expects the train and evaluation dataloaders to a pass images. The End2endModelEvaluator
    expects the premodel to be a feature extractor model that returns representations of the dataloader's images.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        lrs: list[float],
        probe_args: dict,
        seed: int,
        fewshot_k: int,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        train_dataset_config: dict,
        premodel: str | torch.nn.Module | ThingsvisionModel,
        model_dirs: list[str] | None,
        results_dir: str | None,
        normalize: bool = True,
        val_proportion: float = 0,
        force_train: bool = False,
        model_fn: str = "model.pkl",
        reg_lambda_bounds: tuple[int, int] = (-6, 0),
        **kwargs,
    ) -> None:
        if fewshot_k != -1:
            # TODO: should be handeled at a different possition
            raise NotImplementedError(
                "End-to-end probe evaluation is not possible at the moment in fewshot setting. Please set to -1."
            )

        self.device = probe_args.get("device", "cpu")

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, please run on a machine with a GPU.")

        probe_args = self._check_probe_arguments(probe_args, premodel, normalize)

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            lrs=lrs,
            probe_args=probe_args,
            seed=seed,
            fewshot_k=fewshot_k,
            model_dirs=model_dirs,
            results_dir=results_dir,
            normalize=normalize,
            val_proportion=val_proportion,
            force_train=force_train,
            model_fn=model_fn,
            reg_lambda_bounds=reg_lambda_bounds,
        )

        self.probe_fn = check_single_instance(self.probe_fns, "probe filename")

        if eval_dataloader is None:
            raise ValueError(f"Cannot evaluate a trained probe as passed eval_dataloader is None")

        if train_dataloader is None and self.force_train:
            raise ValueError(f"Cannot train a probe ({self.force_train=}), as passed train_dataloader is None")

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_dataset_config = train_dataset_config

    def _check_probe_arguments(
        self,
        probe_args: dict,
        premodel: str | Path | torch.nn.Module | ThingsvisionModel,
        normalize: bool,
    ):
        passed_premodel = probe_args.get("premodel")
        if passed_premodel is None:
            probe_args["premodel"] = premodel
        elif passed_premodel != premodel:
            raise ValueError(
                f"Conflicting arguments! Premodel passed to the probe arguments (`probe_args`) does not match the premodel passed to the End2endModelEvaluator."
            )
        return probe_args

    def _check_and_add_shuffle_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataloader.dataset
        last_op = str(dataset.pipeline[-1])
        if "<_shuffle" not in last_op:
            logger.info(f"Recreating train dataloader with shuffling ...")
            dataset = dataset.shuffle(min(len(dataset), 10_000))
            return DataLoader(
                dataset=dataset,
                batch_size=self.train_dataloader.batch_size,
                shuffle=False,
                num_workers=self.train_dataloader.num_workers,
                pin_memory=self.train_dataloader.pin_memory,
                collate_fn=self.train_dataloader.collate_fn,
            )
        else:
            return self.train_dataloader

    def evaluate(self) -> dict:
        """TODO: write comprehensive docstring"""
        probe_exists = Path(self.probe_fn).exists() and (not self.force_train)

        if probe_exists:
            logger.info(f"Found existing pretrained probe at {self.probe_fn=}")
            self.reg_lambda = None
            hopt_time_s = None
            best_val_bal_acc1 = None
            self.probe.premodel = None
        else:
            logger.info(f"Could not find pretrained probe at {self.probe_fn=}")
            logger.info("Optimizing hyperparameters of linear probe ...")
            st = time.time()
            best_val_bal_acc1 = self.optimize_hyperparams(
                self.train_dataloader, train_dataset_config=self.train_dataset_config
            )
            et = time.time()
            hopt_time_s = et - st

        if isinstance(self.train_dataloader.dataset, wds.WebDataset):
            self.train_dataloader = self._check_and_add_shuffle_train_dataloader()

        metric_dict = self._evaluate(
            train_loader=self.train_dataloader,
            test_loader=self.eval_dataloader,
            filename=self.probe_fn,
            evaluate_train=not probe_exists,
        )
        metric_dict["hopt_time_s"] = hopt_time_s
        metric_dict["best_val_bal_acc1"] = best_val_bal_acc1
        return metric_dict
