from functools import partial
from typing import List

import torch
from loguru import logger
from sklearn.metrics import balanced_accuracy_score, classification_report
from thingsvision.core.rsa import compute_rdm, correlate_rdms

from src.utils.loss_utils import CKATorch, ContrastiveLoss, get_loss


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.

    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.

    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies

    Returns
    -------

    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(k=max(topk), dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]

def balanced_accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,), num_classes: int = None) -> List[float]:
    """
    Compute balanced top-k accuracy (average of per-class accuracies)
    """
    if num_classes is None:
        num_classes = output.shape[1]
    
    pred = output.topk(k=max(topk), dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    # Compute per-class accuracy for each top-k
    balanced_accs = []
    for k in topk:
        class_correct = torch.zeros(num_classes, device=output.device)
        class_counts = torch.zeros(num_classes, device=output.device)
        
        # Count correct predictions and total examples for each class
        for class_id in range(num_classes):
            class_mask = (target == class_id)
            if class_mask.sum() > 0:  # Only process classes that have examples
                class_correct[class_id] = correct[:k][:, class_mask].any(dim=0).sum()
                class_counts[class_id] = class_mask.sum()
        
        # Compute balanced accuracy (average of per-class accuracies)
        valid_classes = class_counts > 0
        if valid_classes.sum() > 0:
            per_class_acc = class_correct[valid_classes] / class_counts[valid_classes]
            balanced_acc = per_class_acc.mean().item()
        else:
            balanced_acc = 0.0
            
        balanced_accs.append(balanced_acc)
    
    return balanced_accs

def compute_metrics(logits: torch.Tensor, target: torch.Tensor, print_report: bool = False, split: str = "test") -> dict:
    pred = logits.argmax(dim=1)

    # measure accuracy
    if target.max() >= 5:
        acc1, acc5 = accuracy(logits.float(), target.float(), topk=(1, 5))
        bal_acc1, bal_acc5 = balanced_accuracy(logits.float(), target.float(), topk=(1, 5))
    else:
        (acc1,) = accuracy(logits.float(), target.float(), topk=(1,))
        acc5 = float("nan")
        (bal_acc1,) = balanced_accuracy(logits.float(), target.float(), topk=(1,))
        bal_acc5 = float("nan")
    
    fair_info = {
        "lp_acc1": acc1,
        "lp_acc5": acc5,
        "lp_bal_acc1": bal_acc1,
        "lp_bal_acc5": bal_acc5,
    }
    if print_report:
        logger.info(f"{split.capitalize()} classification report:")
        logger.info(classification_report(target, pred, digits=3))
    logger.info(f"{split.capitalize()} acc1: {acc1}, bal_acc1: {bal_acc1}")
    return fair_info


class RepresentationMetrics:
    """Compute the metrics for the representation transfer learning task."""

    def __init__(
        self,
        max_samples_similarity: int = 20000,
        contrastive_temperature: float = 0.05,
        device: str = "cuda",
        criterion: str = "mse",
        local_sigma: float = 0.2,
    ) -> None:
        """Initialize the metrics class. We use a maximum number of samples for the similarity metrics such as CKA."""
        self.max_samples_similarity = max_samples_similarity
        # Torch backend compiles, which is slow
        self.cka_global = partial(CKATorch, kernel="linear", unbiased=True, device="cpu", compile=False)
        self.cka_local = partial(
            CKATorch,
            kernel="rbf",
            sigma=local_sigma,
            unbiased=True,
            device="cpu",
            compile=False,
        )
        self.device = device
        self.local_loss = ContrastiveLoss(temperature=contrastive_temperature)
        self.criterion = criterion

    def subset_features(
        self, pred_features: torch.Tensor, target_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(pred_features.shape[0])[: self.max_samples_similarity]
        pred_features = pred_features[indices]
        target_features = target_features[indices]
        return pred_features, target_features

    def get_cka_metrics(self, pred_features: torch.Tensor, target_features: torch.Tensor) -> tuple[float, float]:
        m = pred_features.shape[0]
        cka_local = self.cka_local(m=m).compare(X=pred_features, Y=target_features)
        cka_global = self.cka_global(m=m).compare(X=pred_features, Y=target_features)

        return cka_local.cpu().item(), cka_global.cpu().item()

    def get_rsa_metrics(self, pred_features: torch.Tensor, target_features: torch.Tensor) -> float:
        rdm_features_pred = compute_rdm(pred_features.cpu().numpy(), method="correlation")
        rdm_features_target = compute_rdm(target_features.cpu().numpy(), method="correlation")
        spearman_corr = correlate_rdms(rdm_features_pred, rdm_features_target, correlation="spearman")
        return spearman_corr

    def compute_metrics(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        same_dim: bool = True,
    ) -> dict:
        metrics = {}
        if same_dim:
            mse = torch.nn.functional.mse_loss(pred_features, target_features)
            mae = torch.nn.functional.l1_loss(pred_features, target_features)
            cosine_distance = 1 - torch.nn.functional.cosine_similarity(pred_features, target_features, dim=1).mean()
            metrics["mse"] = mse.cpu().item()
            metrics["mae"] = mae.cpu().item()
            metrics["cosine_distance"] = cosine_distance.cpu().item()
        if pred_features.shape[0] > self.max_samples_similarity:
            # Subsetting for metrics with squared complexity
            pred_features, target_features = self.subset_features(pred_features, target_features)

        if same_dim and "mvae" not in self.criterion:  # The loss for mvae is more complex and not a single criterion
            loss = get_loss(self.criterion, batch_size=pred_features.shape[0], device=self.device)(
                pred_features, target_features
            )
            metrics["loss"] = loss.cpu().item()

        with torch.no_grad():
            # glocal = self.local_loss(pred_features, target_features) # TODO Disabled loss due to limited gain
            # logger.info("Finished Local Loss")
            cka_local, cka_global = self.get_cka_metrics(pred_features, target_features)
            rsa_corr = self.get_rsa_metrics(pred_features, target_features)
            metrics["cka_local"] = cka_local
            metrics["cka_global"] = cka_global
            metrics["spearman_corr"] = rsa_corr

        for k, v in metrics.items():
            logger.info(f"{k}: {v}")
        return metrics
