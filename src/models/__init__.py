import argparse
from collections.abc import Callable

from src.models.activation_combiner import BaseActivationCombiner, get_activation_combiner

from .thingsvision import ThingsvisionModel, load_thingsvision_model

__all__ = ["BaseActivationCombiner", "get_activation_combiner", "load_model"]


def load_model(
    args: argparse.Namespace,
    activation_combiner: BaseActivationCombiner | None = None,
) -> tuple[ThingsvisionModel, Callable]:
    """Load a model."""
    return load_thingsvision_model(
        model_name=args.model,
        source=args.model_source,
        model_parameters=args.model_parameters,
        device=args.device,
        module_names=args.module_names,
        feature_alignment=args.feature_alignment,
        activation_combiner=activation_combiner,
        freeze_model=args.freeze_premodel,
    )
