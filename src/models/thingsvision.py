import copy
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

import thingsvision
import torch
from loguru import logger
from thingsvision import get_extractor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.activation_combiner import BaseActivationCombiner
from src.models.collate_fn import CollateContextManager


def get_activation_custom(extractor: thingsvision.core.extraction.base.BaseExtractor, name: str) -> Callable:
    """Store a copy of the representations for a specific module of the model."""

    def hook(model, input, output) -> None:
        # store a copy of the tensor rather than the tensor itself
        act = output[0] if isinstance(output, tuple) else output
        extractor.activations[name] = act.clone()

    return hook


class ThingsvisionModel(torch.nn.Module):
    """Wrapper class for thingsvision models."""

    def __init__(
        self,
        extractor: thingsvision.core.extraction.base.BaseExtractor,
        module_names: str | list[str],
        token: str | list[str] = "avg_pool",
        flatten_acts: bool = False,
        activation_combiner: BaseActivationCombiner | None = None,
        freeze_model: bool = False,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self._module_names = [module_names] if isinstance(module_names, str) else module_names
        self.extractor.model = self.extractor.model.to(extractor.device)

        self.model = self.extractor.model

        self.extractor.activations = {}
        self.output_type = "tensor"
        self.flatten_acts = flatten_acts
        self.token = self._check_token(token)
        self._original_get_activation = None  # Store original method for pickling
        self._handle_freeze_model(freeze_model)
        logger.info(f"Freezing model: {self.freeze_model}")
        self.extractor._register_hooks(module_names=self._module_names)
        self.activation_combiner = activation_combiner

        logger.info(
            f"Initialized ThingsvisionModel with\n{self._module_names=}\n{self.token=}\n{self.flatten_acts=}\n{self.activation_combiner=}\n{self.freeze_model=}"
        )

    @property
    def module_names(self) -> list[str]:
        """Get the module names of the model."""
        return self._module_names

    def _check_and_register_hooks(self) -> None:
        if not self.extractor.hook_handles:
            logger.info(f"Registering hooks for {self._module_names}")
            if self.freeze_model:
                self._override_extractor_fun()
            self.extractor.activations = {}
            self.extractor._register_hooks(module_names=self._module_names)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        self._check_and_register_hooks()
        _ = self.extractor.forward(batch)
        acts = []
        for module_name, token in zip(self._module_names, self.token, strict=True):
            act = self.extractor.activations[module_name]  # (B, tokens, dim) -> (B, dim | 2*dim)
            acts.append(self._aggregate_rep(act, token))
        if self.activation_combiner is not None:
            acts = self.activation_combiner(acts)
        return acts

    def extract_features(self, batches: DataLoader, output_dir: str, split: str = "_train") -> None:
        """Extract features from model for all batches in a dataloader of images. Features are extracted with the
        thingsvision extractor and saved in the output_dir.

        Args:
            batches: DataLoader of images.
            output_dir: Directory to save the features.
            split: Suffix for the filename of the features.
        """
        self._check_modules_path_exists(output_dir)

        with CollateContextManager(batches) as dataloader:
            self._extract_features(
                batches=dataloader,
                module_names=self._module_names,
                flatten_acts=self.flatten_acts,
                output_type=self.output_type,
                output_dir=output_dir,
                file_name_suffix=split,
                save_in_one_file=True,
            )

    def extract_targets(self, batches: DataLoader, output_dir: str, split: str = "train") -> None:
        """Extract targets from a batch of images."""
        new_fn_path = Path(output_dir) / f"targets{split}.pt"
        if new_fn_path.exists():
            logger.warning(f"Targets file already exists at {new_fn_path}")
        else:
            targets = torch.cat([target for _, target in batches])
            torch.save(targets, new_fn_path)

        for module_name in self._module_names:
            module_dir = Path(output_dir) / module_name
            module_dir.mkdir(parents=True, exist_ok=True)
            symlink_path = module_dir / f"targets{split}.pt"
            try:
                if symlink_path.exists():
                    logger.warning(f"Extracted targets file already exists at {symlink_path}. Skipping.")
                    continue
                elif symlink_path.is_symlink():
                    logger.warning(f"Extracted targets file is a symlink at {symlink_path}. Removing it.")
                    symlink_path.unlink()
                symlink_path.symlink_to(new_fn_path.resolve())
            except Exception as e:
                logger.warning(f"Could not create symlink {symlink_path} -> {new_fn_path}: {e}")
                raise e

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        """Get the parameters of the model."""
        return self.extractor.model.parameters()

    def n_parameters(self) -> int:
        """Get the number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    def _handle_freeze_model(self, freeze_model: bool) -> None:
        if freeze_model:
            for param in self.extractor.model.parameters():
                param.requires_grad = False
            self._initial_state_dict = None
        else:
            self._override_extractor_fun()
            for param in self.extractor.model.parameters():
                param.requires_grad = True
            logger.info(f"Saving initial state dict of the model")
            self._initial_state_dict = copy.deepcopy(self.extractor.model.state_dict())
        self.freeze_model = freeze_model

    def reset_to_pretrained(self) -> None:
        """Reset model to original pretrained weights."""
        if self._initial_state_dict is None:
            logger.warning("No initial state saved. Cannot reset.")
            return
        self.extractor.model.load_state_dict(self._initial_state_dict)
        logger.info("Reset model to original pretrained weights")

    def _check_token(self, token: str | list[str]) -> list[str]:
        if not isinstance(token, (list, str)):
            raise ValueError(f"Passed {token=} is not a string or list of strings.")
        if isinstance(token, str):
            token = [token]
        for t in token:
            if t not in ["cls_token", "avg_pool", "cls_token+avg_pool", "all_tokens"]:
                raise ValueError(
                    f"Unknown {t=}. Allowed ones: ['cls_token', 'avg_pool', 'cls_token+avg_pool', 'all_tokens']"
                )
        if len(token) == 1:
            return token * len(self._module_names)
        elif len(token) == len(self._module_names):
            return token
        else:
            raise ValueError(
                f"Passed {token=} is a list of strings with length {len(token)} but expected {len(self._module_names)}. "
                "If the same token should be used for all modules, pass a single string."
            )

    def _override_extractor_fun(self) -> None:
        """Override the extractor to remove .detach()"""
        # Store original method if not already stored
        if self._original_get_activation is None:
            self._original_get_activation = getattr(self.extractor, "get_activation", None)

        def new_get_activation(name: str) -> Callable:
            logger.info(f"Registering hook for {name} with the custom function")

            def hook(model, input, output) -> None:
                act = output[0] if isinstance(output, tuple) else output
                self.extractor.activations[name] = act.clone()  # No .detach()!

            return hook

        logger.info(f"Overriding extractor.get_activation with the custom function")
        self.extractor.get_activation = new_get_activation

    def _aggregate_rep(self, act: torch.Tensor, token: str) -> torch.Tensor:
        # in the real code they use "clone" everywhere not sure we need it as well
        if token == "all_tokens":
            return act
        elif token == "cls_token":
            return act[:, 0, :]
        elif token == "avg_pool":
            return act[:, 1:, :].mean(dim=1)
        else:
            cls_token = act[:, 0, :]
            pooled_tokens = act[:, 1:, :].mean(dim=1)
            return torch.stack((cls_token, pooled_tokens), dim=1)

    def __getstate__(self) -> dict:
        """Called before pickling. Remove hooks and restore original get_activation to avoid pickling errors."""
        state = self.__dict__.copy()

        if hasattr(self, "_modules") and self._modules:
            logger.info(f"Saving modules: {self._modules.keys()}")
            # state['_saved_modules'] = {k: v for k, v in self._modules.items()}
            state["_saved_modules"] = {"activation_combiner": self._modules["activation_combiner"]}

        for attr in [
            "_modules",
            "_parameters",
            "_buffers",
            "_non_persistent_buffers_set",
            "_backward_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
        ]:
            state.pop(attr, None)

        try:
            logger.info(f"Unregistering hooks during pickling")
            self.extractor._unregister_hooks()
            self.extractor.activations = {}
            state["extractor"].activations = {}

            if hasattr(self.extractor, "get_activation") and self._original_get_activation is not None:
                state["extractor"].get_activation = self._original_get_activation
                self.extractor.get_activation = self._original_get_activation
        except Exception as e:
            logger.warning(f"Could not unregister hooks during pickling: {e}")
            raise e
        return state

    def __setstate__(self, state: dict) -> None:
        """Called after unpickling. Restore hooks."""
        logger.info(f"Restoring state of ThingsvisionModel - __setstate__ called")

        torch.nn.Module.__init__(self)

        saved_modules = state.pop("_saved_modules", {})
        self.__dict__.update(state)

        for name, module in saved_modules.items():
            setattr(self, name, module)

        self.model = self.extractor.model

        self.extractor.activations = {}
        try:
            if not self.freeze_model:
                self._override_extractor_fun()
            self.extractor._register_hooks(module_names=self._module_names)
        except Exception as e:
            logger.warning(f"Could not restore hooks after unpickling: {e}")
            raise e

        logger.info(f"{hasattr(self, '_initial_state_dict')=}")

    def _check_modules_path_exists(self, output_dir: str) -> None:
        """Check if the model path exists."""
        for module_name in self._module_names:
            path = Path(output_dir) / module_name
            logger.info(path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Module path does not exist: {path}. Creating it.")

    def _extract_features(
        self,
        batches: DataLoader,
        module_names: list[str] | None = None,
        flatten_acts: bool = False,
        output_type: str = "tensor",
        output_dir: str | None = None,
        step_size: int | None = None,
        file_name_suffix: str = "",
        save_in_one_file: bool = False,
    ) -> None:
        self.extractor.model = self.extractor.model.to(self.extractor.device)
        self.extractor.activations = {}

        if module_names is None:
            module_names = self._module_names

        self.extractor._module_and_output_check(module_names, output_type)

        if output_dir is None:
            raise ValueError("output_dir is required")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if not step_size:
            step_size = 8000 // (len(next(iter(batches))) * 3) + 1

        features = defaultdict(list)
        feature_file_names = defaultdict(list)
        image_ct, last_image_ct = 0, 0
        for i, batch in tqdm(enumerate(batches, start=1), desc="Batch", total=len(batches)):
            modules_features = self.extractor._extract_batch(
                batch=batch, module_names=module_names, flatten_acts=flatten_acts
            )

            image_ct += len(batch)
            del batch

            for module_name in module_names:
                features[module_name].append(modules_features[module_name])

                if i % step_size == 0 or i == len(batches):
                    features_subset = torch.cat(features[module_name])
                    features_subset_file = (
                        Path(output_dir) / f"{module_name}/features{file_name_suffix}_{last_image_ct}-{image_ct}.pt"
                    )
                    torch.save(features_subset, features_subset_file)

                    features[module_name] = []
                    last_image_ct = image_ct
                    feature_file_names[module_name].append(features_subset_file)

        if save_in_one_file:
            for module_name in module_names:
                all_features = [torch.load(file) for file in feature_file_names[module_name]]
                features_file = Path(output_dir) / f"{module_name}/features{file_name_suffix}.pt"
                torch.save(torch.cat(all_features), features_file)
                for file in feature_file_names[module_name]:
                    file.unlink()


def argument_processing(
    model_name: str | list[str],
    source: str | list[str],
    model_parameters: dict | list[dict],
    module_names: str | list[str],
) -> tuple[str, str, dict, list[str], str | list[str] | None, list[str], bool]:
    """Process the arguments for the model."""
    ## Case using thingsvision model for feature extraction
    if isinstance(model_name, str) and isinstance(source, str) and isinstance(model_parameters, dict):
        flatten_acts = True
        if model_parameters.get("extract_cls_token") or "token_extraction" in model_parameters:
            flatten_acts = False
            curr_token = model_parameters.get("token_extraction")
            if curr_token == "all_tokens":
                model_parameters = model_parameters.copy()
                model_parameters.pop("token_extraction")
            tokens_list = [curr_token]
    ## Case using thingsvision model for end2end training
    elif isinstance(model_name, list) and isinstance(source, list) and (isinstance(model_parameters, (list, dict))):
        if isinstance(model_parameters, dict):
            model_parameters = [model_parameters] * len(model_name)

        if not len(model_name) == len(source) == len(module_names) == len(model_parameters):
            raise ValueError(f"Invalid arguments: {model_name=}, {source=}, {model_parameters=}, {module_names=}")

        if len(set(model_name)) != 1 or len(set(source)) != 1:
            raise ValueError(
                f"model_name and source must each have only one unique value for end2end training. Got: {model_name=}, {source=}"
            )

        flatten_acts = False  # by default we don't want thingsvision to flatten the activations for end2end training
        tokens_list = [model_param.get("token_extraction", "avg_pool") for model_param in model_parameters]
        assert len(tokens_list) == len(module_names), (
            f"len(tokens_list)={len(tokens_list)}, len(module_names)={len(module_names)}"
        )
        # prepare final model parameters for the first module
        model_name = model_name[0]
        source = source[0]
        model_parameters = model_parameters[0]
        if "token_extraction" in model_parameters:
            model_parameters = model_parameters.copy()
            model_parameters.pop("token_extraction")
    else:
        raise TypeError(
            f"Invalid arguments: {model_name=}, {source=}, {model_parameters=}. Expected single string and "
            "dict for model_name, source, and model_parameters. If multiple modules are loaded for end2end "
            f"training, provide a list of strings and dicts for model_name, source, and model_parameters. "
            f"Got {type(model_name)=}, {type(source)=}, {type(model_parameters)=}."
        )
    return model_name, source, model_parameters, module_names, tokens_list, flatten_acts


def load_thingsvision_model(
    model_name: str | list[str],
    source: str | list[str],
    device: str | torch.device,
    model_parameters: dict | list[dict],
    module_names: str | list[str],
    feature_alignment: str | list[str] | None = None,
    activation_combiner: BaseActivationCombiner | None = None,
    freeze_model: bool = False,
) -> tuple[ThingsvisionModel, Callable]:
    """Loads a thingsvision extractor/model along with its associated transformation function.
    Only a single model can be loaded at a time; if `model_name` or `source` are provided as lists,
    all entries must be identical. The ThingsvisionModel will register hooks for the modules specified
    in `module_names` and aggregate their representations according to the strategy defined
    in `model_parameters`. If `model_parameters` is a dict, then the model will use the same
    parameters for all modules. If `model_parameters` is a list of dicts, then the model will use the
    module-specific parameters for each module.

    Basically, we have the setting:
    1. If `model_name` and `source` are strings, and `model_parameters` is a dict, then we use the same parameters for
       all modules and the model should ONLY be used for feature extraction.
    2. If `model_name` and `source` are list of strings, and `model_parameters` is a list of dicts, then we use the
       module-specific parameters for each module and the model SHOULD be used for end2end training.
    """
    (
        model_name,
        source,
        model_parameters,
        module_names,
        tokens_list,
        flatten_acts,
    ) = argument_processing(
        model_name=model_name,
        source=source,
        model_parameters=model_parameters,
        module_names=module_names,
    )
    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True,
        model_parameters=model_parameters,
    )

    model = ThingsvisionModel(
        extractor=extractor,
        module_names=module_names,
        token=tokens_list,
        flatten_acts=flatten_acts,
        activation_combiner=activation_combiner,
        freeze_model=freeze_model,
    )
    transform = extractor.get_transformations(resize_dim=256, crop_dim=224)
    return model, transform
