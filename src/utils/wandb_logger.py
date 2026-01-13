from typing import Any, Dict, Optional

import torch.nn as nn
import wandb


class WandBLogger:
    """Wrapper around the wandb library."""

    def __init__(
        self,
        project: str = "rep2rep",
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """Initialize the WandBLogger.

        Args:
            project (str): The name of the project.
            config (Optional[Dict[str, Any]]): The configuration for the experiment.
            enabled (bool): Whether to enable the logger. Defaults to True.
        """
        self.enabled = enabled
        if not enabled:
            return
        self.config = config or {}
        wandb.init(project=project, config=self.config)

    def watch_model(self, model: nn.Module) -> None:
        """Start gradient and parameter logging for a model."""
        if self.enabled:
            wandb.watch(model)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics for the current step."""
        if self.enabled:
            wandb.log(metrics)

    def log_summary(self, metrics: Dict[str, float]) -> None:
        """Log final summary metrics."""
        if self.enabled:
            for key, value in metrics.items():
                wandb.run.summary[key] = value

    def save_model(self, model_path: str) -> None:
        """Save model as a W&B artifact."""
        if self.enabled:
            wandb.save(model_path)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled:
            wandb.finish()
