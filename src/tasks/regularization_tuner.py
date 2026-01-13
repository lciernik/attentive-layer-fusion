import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from src.eval.metrics import balanced_accuracy
from src.tasks.attentive_probe import AttentiveProbe
from src.tasks.linear_probe import LinearProbe


class RegularizationTuner:
    def __init__(
        self,
        probe: LinearProbe | AttentiveProbe,
        lr: float,
    ):
        self.probe = probe
        self.lr = lr

    def find_peak(
        self,
        lambda_list: list[float],
        idxs: list[int],
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[int, float]:
        best_lambda_idx, max_acc = 0, 0
        for idx in idxs:
            reg_lambda = lambda_list[idx]
            self.probe.reinit_model(
                train_loader,
                params_to_set={"reg_lambda": reg_lambda, "lr": self.lr},
            )
            self.probe.train(train_loader)
            logits, target = self.probe.infer(val_loader)
            (acc1,) = balanced_accuracy(logits.float(), target.float(), topk=(1,))
            logger.info(f"Validation accuracy AFTER training with regularization lambda {reg_lambda}: {acc1}")
            if max_acc < acc1:
                best_lambda_idx, max_acc = idx, acc1
        return best_lambda_idx, max_acc

    def tune_lambda(
        self,
        feature_train_loader: DataLoader,
        feature_val_loader: DataLoader,
        min_exp: int = -6,
        max_exp: int = 0,
        vals_between_init: int = 0,
    ) -> tuple[float, float]:
        """
        Perform openAI-like hyperparameter sweep
        https://arxiv.org/pdf/2103.00020.pdf A.3
        instead of scikit-learn LBFGS use FCNNs with AdamW
        For an initial set of lambda values, we find the best lambda value.
        Between each broad lambda value we get 8 fine-grained lambda values.
        We perform a binary search between the best lambda value and the next best lambda value.
        We continue this process until the step_span (beginning with 8) is 0.
        We return the best lambda value and the best accuracy.

        Args:
            feature_train_loader: DataLoader for training features
            feature_val_loader: DataLoader for validation features
            min_exp: Minimum exponent for lambda values, in np.logspace(min_exp, max_exp, num=num_init)
            max_exp: Maximum exponent for lambda values, in np.logspace(min_exp, max_exp, num=num_init)
            num_init: Number of initial lambda values
            vals_between_init: Number of values between each initial lambda value. 

        Returns:
            best_lambda: Best lambda value
            best_acc: Best accuracy
        """

        num_init = max_exp - min_exp + 1
        lambda_list_init = np.logspace(min_exp, max_exp, num=num_init).tolist()
        logger.info(f"lambda_list_init: {lambda_list_init}")
        # Put 8 values between each lambda_list_init value
        lambda_list = np.logspace(min_exp, max_exp, num=(num_init + (num_init - 1) * vals_between_init)).tolist()
        lambda_init_idx = [lambda_list.index(val) for val in lambda_list_init]

        peak_idx, acc1 = self.find_peak(lambda_list, lambda_init_idx, feature_train_loader, feature_val_loader)
        while vals_between_init > 0:
            left, right = (
                max(peak_idx - vals_between_init, 0),
                min(peak_idx + vals_between_init, len(lambda_list) - 1),
            )
            logger.info(f"step_span: {vals_between_init}, left_idx: {left}, right_idx: {right}, peak_idx: {peak_idx}")
            logger.info(f"left_lambda: {lambda_list[left]}, right_lambda: {lambda_list[right]}, current peak: {lambda_list[peak_idx]}")
            # avoid testing the peak_idx
            idxs_to_test = [idx for idx in [left, right] if idx != peak_idx]
            if len(idxs_to_test) == 0:
                break
            new_peak_idx, new_acc1 = self.find_peak(
                lambda_list, idxs_to_test, feature_train_loader, feature_val_loader
            )
            if new_acc1 > acc1:
                acc1 = new_acc1
                peak_idx = new_peak_idx
            vals_between_init //= 2
        best_lambda = lambda_list[peak_idx]
        return best_lambda, acc1
