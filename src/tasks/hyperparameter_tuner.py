from loguru import logger
from torch.utils.data import DataLoader

from src.tasks.attentive_probe import AttentiveProbe
from src.tasks.linear_probe import LinearProbe
from src.tasks.regularization_tuner import RegularizationTuner


class HyperparameterTuner:
    """This class is used to find the best hyperparameter setting (learning rate and regularizaation parameter) for the
    linear probe.  It uses the RegularizationTuner to find the best regularization value for each combination of the
    other hyperparameters.
    """

    def __init__(
        self,
        lrs: list[float],
        probe: LinearProbe | AttentiveProbe,
        max_epochs_for_tuning: int| None = 40,
    ) -> None:
        self.lrs = lrs
        self.probe = probe
        self.max_epochs_for_tuning = max_epochs_for_tuning

    def tune(
        self,
        feature_train_loader: DataLoader,
        feature_val_loader: DataLoader,
        min_exp: int = -6,
        max_exp: int = 0,
        vals_between_init: int = 0,
    ) -> tuple[float, float, float]:
        """Tune the hyperparameters."""
        if self.max_epochs_for_tuning:
            original_n_epochs = self.probe.epochs
            self.probe.set_epochs(self.max_epochs_for_tuning)

        best_lr, best_reg_lambda, max_acc = 0, 0, 0
        logger.info("-" * 100)
        logger.info("-" * 100)
        logger.info(f"Tuning hyperparameters with lrs: {self.lrs}")
        logger.info("-" * 100) 
        for lr in self.lrs:
            logger.info(f"Starting tuning with lr: {lr}")
            logger.info("-" * 100)
            regularization_tuner = RegularizationTuner(self.probe, lr)
            reg_lambda, acc1 = regularization_tuner.tune_lambda(
                feature_train_loader,
                feature_val_loader,
                min_exp,
                max_exp,
                vals_between_init,
            )
            logger.info(f"End of tuning with lr {lr} and reg_lambda {reg_lambda}: {acc1}")
            logger.info("-" * 100)
            logger.info("-" * 100)
            if max_acc < acc1:
                best_lr, best_reg_lambda, max_acc = lr, reg_lambda, acc1

        if self.max_epochs_for_tuning:
            self.probe.set_epochs(original_n_epochs)

        return best_lr, best_reg_lambda, max_acc
