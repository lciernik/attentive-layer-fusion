import argparse
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger

from src.utils.model_mapping import compress_consecutive_sequences, module_shortener, split_model_module
from src.utils.tasks import Task
from src.utils.utils import all_paths_exist, as_list


class PathMaker:
    """Class to make all project paths, i.e., paths to store features, models and results."""

    def __init__(
        self,
        args: argparse.Namespace,
        dataset_name: str,
        probe_dataset_name: str | None = None,
        auto_create_dirs: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.train_dataset_name = probe_dataset_name if probe_dataset_name is not None else dataset_name
        self.task = args.task
        self.mode = args.mode

        self.auto_create_dirs = auto_create_dirs

        self.dataset_root = args.dataset_root
        self.feature_root = args.feature_root
        self.model_root = args.model_root
        self.output_root = args.output_root
        self.freeze_premodel = args.freeze_premodel
        self._check_root_paths()

        self.model_ids = as_list(args.model_key)
        self.feature_combiner = args.feature_combiner
        self.fewshot_slug = "no_fewshot" if args.fewshot_k == -1 else f"fewshot_{args.fewshot_k}"
        self.hyperparams_slug = self._get_hyperparams_name(args)
        self.clustering_slug = self._get_clustering_slug(args)
        self.model_slug = self._create_model_slug()

        self.verbose = args.verbose

    def _get_clustering_slug(self, args: argparse.Namespace) -> str:
        if self.task in {Task.LINEAR_PROBE, Task.ATTENTIVE_PROBE} and self.mode == "combined_models":
            return f"sim_{args.clustering_similarity_method}/num_clusters_{args.num_clusters}"
        else:
            return None

    def _get_hyperparams_name(self, args: argparse.Namespace) -> str:
        """Get the hyperparameters name for the output path."""
        path_elements = [
            f"grad_norm_clip_{args.grad_norm_clip}" if args.grad_norm_clip is not None else "no_grad_norm_clip",
            f"jitter_p_{args.jitter_p}",
            self.fewshot_slug,
            f"regularization_{args.regularization}",
            f"seed_{args.seed}",
        ]

        if args.task == Task.SAE_TRAINING:
            path_elements = [
                f"k_{args.k if args.k is not None else 'auto'}",
                f"increase_factor_{args.increase_factor}",
                f"loss_type_{args.loss_type}",
                f"val_prop_{args.val_proportion}",
                *path_elements,
            ]
        elif args.task == Task.REP2REP or (args.task == Task.LINEAR_PROBE and args.mode == "mvae_eval"):
            path_elements = [
                f"loss_{args.rep_loss}",
                f"regularization_{args.regularization}",
                "normalized" if args.normalize else "not_normalized",
                f"seed_{args.seed}",
            ]
        elif args.task == Task.ATTENTIVE_PROBE:
            alignment_slug = f"dim_align_{args.dimension_alignment}"

            if args.dimension_alignment == "linear_projection":
                if args.always_project:
                    alignment_slug += "_all"
                else:
                    alignment_slug += "_on_mismatch"

            path_elements = [
                f"dim_{args.dim}",
                f"num_heads_{args.num_heads}",
                alignment_slug,
                f"attn_drop_{args.attention_dropout[0]}_{args.attention_dropout[1]}",
                *path_elements,
            ]
        subpath = Path().joinpath(*path_elements)
        return str(subpath)

    def _collapse_model_ids(self, model_ids: list[str]) -> str:
        pp_model_ids = [split_model_module(model_id) for model_id in model_ids]
        if pp_model_ids[0][0].endswith("_at"):
            if not all(pp_model_ids[0][0] == curr_comb[0] for curr_comb in pp_model_ids):
                mismatched = [curr_comb[0] for curr_comb in pp_model_ids if curr_comb[0] != pp_model_ids[0][0]]
                raise ValueError(
                    f"Cannot collapse model ids: not all model ids end with the same '_at' model prefix.\n"
                    f"Expected all to be '{pp_model_ids[0][0]}', but got mismatches: {mismatched}\n"
                    f"Full model ids: {model_ids}\n"
                )
            logger.warning(
                f"Assume that all modules have been passed as modules of the last layer, "
                f"therefore we only use the model_id as the model_slug."
            )
            return pp_model_ids[0][0]

        groups = defaultdict(list)
        for mid, module_name in pp_model_ids:
            new_module_name = module_name
            if module_name is not None:
                new_module_name = module_shortener(new_module_name)
            groups[mid].append(new_module_name)
        parts = []
        for prefix, module_names in groups.items():
            if module_names == [None]:
                parts.append(prefix)
            elif len(module_names) == 1:
                parts.append(f"{prefix}@{module_names[0]}")
            else:
                clean_suffixes = [s for s in module_names if s is not None]
                if len(clean_suffixes) != len(module_names):
                    raise ValueError(f"Somehow a model is missing its module: {module_names}.")
                clean_suffixes = compress_consecutive_sequences(clean_suffixes)
                parts.append(f"{prefix}@{'-'.join(clean_suffixes)}")
        return "__".join(parts)

    def _create_model_slug(self, model_ids: list[str] | None = None) -> str:
        if model_ids is None:
            model_ids = self.model_ids

        model_slug = model_ids[0] if len(model_ids) == 1 else self._collapse_model_ids(model_ids)

        if self.task in {Task.LINEAR_PROBE, Task.ATTENTIVE_PROBE} and self.mode in ["combined_models", "end_2_end"]:
            model_slug += f"_{self.feature_combiner}"
            if self.mode == "end_2_end":
                model_slug += "_frozen_premodel" if self.freeze_premodel else "_unfrozen_premodel"
        elif self.task == Task.SAE_TRAINING:
            model_slug += "_sae"
        return model_slug

    def _check_root_paths(self) -> None:
        """Check existence of the feature, model and output folders."""
        if not Path(self.dataset_root).exists():
            raise FileNotFoundError(f"Dataset root folder {self.dataset_root} does not exist.")
        if not Path(self.feature_root).exists():
            raise FileNotFoundError(f"Feature root folder {self.feature_root} does not exist.")
        if not Path(self.model_root).exists() and self.task == Task.LINEAR_PROBE:
            raise FileNotFoundError(f"Model root folder {self.model_root} does not exist.")
        if not Path(self.output_root).exists() and self.task == Task.LINEAR_PROBE and self.auto_create_dirs:
            Path(self.output_root).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created path ({self.output_root}), where results are to be stored ...")

    def _get_feature_dirs(self) -> list[str]:
        if self.mode == "end_2_end":
            return []

        fmt_model_ids = [
            "/".join(split_model_module(model_id, use_abbrev=False, always_return_tuple=False))
            for model_id in self.model_ids
        ]

        feature_dirs = [str(Path(self.feature_root) / self.dataset_name / model_id) for model_id in fmt_model_ids]

        if self.task == Task.LINEAR_PROBE and self.mode != "single_model" and not all_paths_exist(feature_dirs):
            raise FileNotFoundError(
                f"Not all feature directories exist: {feature_dirs}. "
                f"Cannot evaluate linear probe with multiple models."
                f"Run the linear probe for each model separately first."
            )
        return feature_dirs

    def _get_model_dirs(self) -> list[str]:
        if self.task in {Task.LINEAR_PROBE, Task.ATTENTIVE_PROBE} and self.mode in ["combined_models", "end_2_end"]:
            model_dirs = [
                str(Path(self.model_root) / self.train_dataset_name / self.model_slug / self.hyperparams_slug)
            ]
        else:
            model_dirs = [
                str(
                    Path(self.model_root)
                    / self.train_dataset_name
                    / self._create_model_slug([model_id])
                    / self.hyperparams_slug
                )
                for model_id in self.model_ids
            ]
        return model_dirs

    def _get_results_dirs(self) -> str:
        base_results_dir = self.get_base_results_dir()
        results_dir = Path(base_results_dir) / self.hyperparams_slug

        if self.task == Task.LINEAR_PROBE and self.mode == "mvae_eval":
            results_dir = results_dir / self.dataset_name / self.fewshot_slug

        if not results_dir.exists() and self.auto_create_dirs:
            results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created path ({results_dir}), where results are to be stored ...")

        return str(results_dir)

    def get_base_results_dir(self) -> str:
        """Get the base results directory."""
        if self.mode == "mvae_eval":
            base_results_dir = Path(self.output_root) / "rep2rep" / "mvae" / self.train_dataset_name / self.model_slug
        else:
            prefix_path = Path(self.output_root) / self.task.value / self.mode / self.dataset_name
            if self.clustering_slug is not None:
                prefix_path = prefix_path / self.clustering_slug
            base_results_dir = prefix_path / self.model_slug
        return str(base_results_dir)

    def _get_single_prediction_dirs(self) -> list[str]:
        single_prediction_dirs = [
            str(
                Path(self.output_root)
                / self.task.value
                / "single_model"
                / self.dataset_name
                / model_id
                / self.hyperparams_slug
            )
            for model_id in self.model_ids
        ]
        if not all_paths_exist(single_prediction_dirs):
            raise FileNotFoundError(
                f"Not all single prediction directories exist: {single_prediction_dirs}. "
                f"Cannot evaluate ensemble model."
            )
        return single_prediction_dirs

    def make_paths(
        self,
    ) -> tuple[list[str], list[str] | None, str | None, list[str] | None, list[str]]:
        """Make all project paths, i.e., paths to store features, models and results.

        Returns:
            feature_dirs: List of feature directories.
            model_dirs: List of model directories.
            results_dir: Path to the results directory.
            single_prediction_dirs: List of single prediction directories.
            model_ids: List of model ids.
        """
        feature_dirs = self._get_feature_dirs()

        if self.task == Task.FEATURE_EXTRACTION:
            return feature_dirs, None, None, None, self.model_ids

        model_dirs = self._get_model_dirs()

        results_dir = self._get_results_dirs()

        if self.task == Task.LINEAR_PROBE and self.mode == "ensemble":
            single_prediction_dirs = self._get_single_prediction_dirs()
        else:
            single_prediction_dirs = None

        for d in [
            ("feature_dirs", feature_dirs),
            ("model_dirs", model_dirs),
            ("results_dir", results_dir),
            ("single_prediction_dirs", single_prediction_dirs),
            ("model_ids", self.model_ids),
        ]:
            if isinstance(d[1], list):
                logger.info(f"{d[0]}: {json.dumps(d[1], indent=2)}")
            else:
                logger.info(f"{d[0]}: {d[1]}")

        return (
            feature_dirs,
            model_dirs,
            results_dir,
            single_prediction_dirs,
            self.model_ids,
        )
