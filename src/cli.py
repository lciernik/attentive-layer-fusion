import json
import os
import random
import sys
import traceback
from copy import copy
from itertools import product
from pathlib import Path

from loguru import logger

import src.tasks as tasks
from src.argparser import (
    get_parser_args,
    load_model_configs_args,
    prepare_args,
    prepare_combined_args,
)
from src.cluster_models import get_model_combinations_from_clustering
from src.data import get_dataset_class_filter, get_feature_combiner_cls, get_image_dataloader, get_list_of_datasets
from src.models import get_activation_combiner, load_model
from src.utils.path_maker import PathMaker
from src.utils.tasks import Task
from src.utils.utils import (
    as_list,
    check_existing_results,
    get_base_evaluator_args,
    get_combination,
    get_list_of_models,
    prepare_device,
    reduce_list_of_models,
    save_results,
    set_all_random_seeds,
    world_info_from_env,
)
from src.data.data_utils import map_to_probe_dataset, check_force_train, prepare_ds_name


def main():
    parser, base = get_parser_args()
    base = load_model_configs_args(base)

    try:
        if base.task == Task.MODEL_SIMILARITY:
            main_model_sim(base)
        else:
            main_eval(base)
    except Exception as e:
        logger.error(f"An error occurred during the run with models {base.model_key}: \n  {e}")
        traceback.print_exc()

        with open(os.path.join(base.output_root, "failed_models.txt"), "a") as f:
            array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "unknown")
            task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "unknown")
            f.write(f"{base.model_key} LOGID {array_job_id}_{task_id} \n")
            f.write(f"{str(e)}\n")


def main_model_sim(base):
    base.device = prepare_device(base.distributed)
    logger.info(f"Using device: {base.device}")

    logger.info("Get list of data to evaluate on")
    datasets = get_list_of_datasets(base)

    dataset = datasets[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    dataset_name = prepare_ds_name(dataset)

    train_split = base.train_split

    if dataset_name == "wds_imagenet1k":
        raise ValueError(
            f"Cannot compute the representational similarity matrix on entire ImageNet. Please subset the datasets first with `scripts/in_subset_extraction.py` and pass new dataset name."
        )
    elif dataset_name in ["wds_imagenetv2", "wds_imagenet-r", "wds_imagenet-a", "wds_imagenet_sketch"]:
        raise ValueError(
            f"Cannot compute the representational similarity matrix on ImageNet OOD datasets. Please compute the representational similarity for ImageNet subsets."
        )
    model_ids = as_list(base.model_key)
    logger.info("Load paths")
    feature_root = Path(base.feature_root) / dataset_name
    subset_root = Path(base.subset_root) / dataset_name if base.use_ds_subset else None
    logger.info("Load metric object")
    sim_mat, model_ids, method_slug = tasks.compute_sim_matrix(
        sim_method=base.sim_method,
        feature_root=feature_root,
        model_ids=model_ids,
        split=train_split,
        subset_root=subset_root,
        kernel=base.sim_kernel,
        rsa_method=base.rsa_method,
        corr_method=base.corr_method,
        backend="torch",
        unbiased=base.unbiased,
        device=base.device,
        sigma=base.sigma,
        num_workers=base.num_workers,
        normalize=base.normalize,
        save_base_path=Path(base.output_root) / dataset_name,
    )

    return 0


def main_eval(base):
    # prepare run combinations
    (fewshot_k, epochs, rnd_seed, regularization), task_id = get_combination(
        base.fewshot_k,
        base.epochs,
        base.seed,
        base.regularization,
    )
    base.fewshot_k = fewshot_k
    base.epochs = epochs
    base.seed = rnd_seed
    base.regularization = regularization
    base.task_id = task_id

    # Get list of models to evaluate
    models = get_list_of_models(base)

    # Get list of data to evaluate on
    datasets = get_list_of_datasets(base)

    logger.info(f"\nModels: {models}")
    logger.info(f"Datasets: {datasets}\n")

    if base.mode == "single_model":
        runs = product(models, datasets)
        arg_fn = prepare_args
    else:
        # All other modes are combinations of models
        model_combinations = [models]
        runs = product(model_combinations, datasets)
        arg_fn = prepare_combined_args

    if base.distributed:
        local_rank, rank, world_size = world_info_from_env()
        runs = list(runs)
        random.seed(base.seed)
        random.shuffle(runs)
        runs = [r for i, r in enumerate(runs) if i % world_size == rank]

    for model_info, dataset in runs:
        args = copy(base)
        if (
            args.num_clusters > 0
            and base.mode != "single_model"
            and args.task in [Task.LINEAR_PROBE, Task.ATTENTIVE_PROBE]
        ):
            # If num_clusters is set, we filter the model info to include only the last model in each cluster.
            # TODO maybe allow in the future to also cluster e.g. dino_cls@norm;dino_ap
            if any(["@" in x[5] for x in model_info]):
                raise ValueError("Clustering is not supported for models with multiple modules atm")
            new_model_set = get_model_combinations_from_clustering(
                clustering_root=args.clustering_root,
                dataset=prepare_ds_name(dataset),
                method_key=args.clustering_similarity_method,
                num_clusters=args.num_clusters,
                model_ids=[x[5] for x in model_info],  # Model Keys
            )
            model_info = reduce_list_of_models(model_info, new_model_set)

        args = arg_fn(args, model_info)

        args.dataset = dataset

        try:
            run(args)
        except Exception as e:
            logger.error(
                f"An error occurred during the run with: {model_info} and {dataset}. Continuing with the next run.",
            )
            traceback.print_exc()
            with open(os.path.join(base.output_root, "failed_models.txt"), "a") as f:
                array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "unknown")
                task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "unknown")
                f.write(f"{base.model_key} LOGID {array_job_id}_{task_id} \n")
                f.write(f"{e}\n")


def run(args):
    # device
    args.device = prepare_device(args.distributed)
    logger.info(f"Using device: {args.device}")
    # set seed.
    set_all_random_seeds(args.seed)
    logger.info(f"Using seed: {args.seed}")

    # fix task
    task = args.task
    mode = args.mode
    # prepare dataset name
    dataset_name = prepare_ds_name(args.dataset)
    if args.pretraining_dataset:
        # Optional provide a dataset name for the pretrained model
        logger.info(f"Using pretrained dataset: {args.pretraining_dataset}")
        probe_dataset_name = prepare_ds_name(args.pretraining_dataset)
    else:
        probe_dataset_name = map_to_probe_dataset(dataset_name)
    args.force_train = check_force_train(dataset_name, args.force_train)

    logger.info(f"args.model_root: {args.model_root}")
    path_maker = PathMaker(args, dataset_name, probe_dataset_name)

    dirs = path_maker.make_paths()
    feature_dirs, model_dirs, results_dir, single_prediction_dirs, model_ids = dirs

    if task == Task.LINEAR_PROBE and mode == "mvae_eval":
        # results_dir contains "downstream dataset / fewshot / " so we need to move two directories up
        premodel_filename = os.path.join(os.path.dirname(os.path.dirname(results_dir)), "model.pkl")
        logger.info(f"Running MVAE with {premodel_filename=}")
        model_dirs = [results_dir]

    if args.skip_existing and check_existing_results(results_dir, fn="results.json"):
        logger.info(f"Skipping existing results in {results_dir=}")
        return 0

    if dataset_name.startswith("wds"):
        dataset_root = os.path.join(
            args.dataset_root,
            "wds",
            f"wds_{args.dataset.replace('wds/', '', 1).replace('/', '-')}",
        )
    else:
        dataset_root = args.dataset_root

    logger.info(f"\nRunning '{task.value}' with mode '{mode}' on '{dataset_name}' with the model(s) '{model_ids}'\n")

    base_kwargs = get_base_evaluator_args(args, feature_dirs, model_dirs, results_dir)

    if task == Task.FEATURE_EXTRACTION:
        model, transform = load_model(args, activation_combiner=None)
        train_dataloader, eval_dataloader, _ = get_image_dataloader(args, dataset_root, transform)

        evaluator = tasks.SingleModelEvaluator(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            **base_kwargs,
        )
        logger.info(f"Extracting features for {model_ids} on {dataset_name} and storing them in {feature_dirs} ...")
        extract_train = check_force_train(dataset_name, True)
        evaluator.ensure_feature_availability(check_train=extract_train)

        logger.info(f"Finished feature extraction for {model_ids} on {dataset_name} ...")

    elif task == Task.LINEAR_PROBE:
        base_kwargs["probe_args"]["probe_type"] = "linear"
        base_kwargs["probe_args"]["logit_filter"] = get_dataset_class_filter(args.dataset, args.device)

        if mode == "single_model":
            evaluator = tasks.SingleModelEvaluator(**base_kwargs)

        elif mode == "combined_models" or mode == "mvae_eval":
            feature_combiner_cls = get_feature_combiner_cls(args.feature_combiner)
            if mode == "mvae_eval":
                base_kwargs["probe_args"]["premodel_filename"] = premodel_filename
                base_kwargs["model_fn"] = "CombinedProbe.pkl"

            evaluator = tasks.CombinedModelEvaluator(feature_combiner_cls=feature_combiner_cls, **base_kwargs)
        elif mode == "ensemble":
            evaluator = tasks.EnsembleModelEvaluator(
                model_ids=model_ids,
                single_prediction_dirs=single_prediction_dirs,
                **base_kwargs,
            )
        elif mode == "end_2_end":
            activation_combiner = get_activation_combiner(
                args.feature_combiner, jitter_p=base_kwargs["jitter_p"], normalize=base_kwargs["normalize"]
            )
            model, transform = load_model(args, activation_combiner=activation_combiner)
            train_dataloader, eval_dataloader, train_dataset_config = get_image_dataloader(args, dataset_root, transform)

            evaluator = tasks.End2endModelEvaluator(
                premodel=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                train_dataset_config=train_dataset_config,
                **base_kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. mode should be `single_model`, `combined_models`, or `ensemble`"
            )

        metrics = evaluator.evaluate()

        save_results(args=args, model_ids=model_ids, metrics=metrics, out_path=results_dir)

    elif task == Task.ATTENTIVE_PROBE:
        base_kwargs["probe_args"]["probe_type"] = "attentive"
        base_kwargs["probe_args"]["dim"] = args.dim
        base_kwargs["probe_args"]["num_heads"] = args.num_heads
        base_kwargs["probe_args"]["dimension_alignment"] = args.dimension_alignment
        base_kwargs["probe_args"]["always_project"] = args.always_project
        base_kwargs["probe_args"]["logit_filter"] = get_dataset_class_filter(args.dataset, args.device)
        base_kwargs["probe_args"]["attention_dropout"] = args.attention_dropout

        if mode == "combined_models":
            feature_combiner_cls = get_feature_combiner_cls(args.feature_combiner, shared_dim=args.dim)
            evaluator = tasks.CombinedModelEvaluator(feature_combiner_cls=feature_combiner_cls, **base_kwargs)

        elif mode == "end_2_end":
            activation_combiner = get_activation_combiner(
                args.feature_combiner, shared_dim=args.dim, jitter_p=base_kwargs["jitter_p"], normalize=base_kwargs["normalize"]
            )
            model, transform = load_model(args, activation_combiner=activation_combiner)
            train_dataloader, eval_dataloader, train_dataset_config = get_image_dataloader(args, dataset_root, transform)
            evaluator = tasks.End2endModelEvaluator(
                premodel=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                train_dataset_config=train_dataset_config,
                **base_kwargs,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. mode should be `combined_models`, or  `end_2_end`")

        metrics = evaluator.evaluate()

        save_results(args=args, model_ids=model_ids, metrics=metrics, out_path=results_dir)

    elif task == Task.REP2REP:
        base_kwargs["rep_loss"] = args.rep_loss
        base_kwargs["eval_with_lin_probe"] = args.lin_probe_eval
        if mode == "linear":
            rep_transfer = tasks.LinearRepTransfer(**base_kwargs)
        elif mode == "mvae":
            rep_transfer = tasks.MVAERepTransfer(**base_kwargs)
        else:
            raise NotImplementedError("rep2rep task not implemented")
        metrics = rep_transfer.evaluate()
        logger.info(f"\nmetrics: {json.dumps(metrics, indent=4)}\n")

        save_results(args=args, model_ids=model_ids, metrics=metrics, out_path=results_dir)

    elif task == Task.SAE_TRAINING:
        configs = tasks.get_sae_config_from_args(args)
        evaluator = tasks.SAEEvaluator(**configs)

        metrics = evaluator.evaluate(
            feature_dir=feature_dirs,
            model_dir=model_dirs,
            latent_feature_dir=results_dir,
            extract_train=args.extract_train,
            extract_test=args.extract_test,
        )
        save_results(args=args, model_ids=model_ids, metrics=metrics, out_path=results_dir)
    else:
        raise ValueError(f"Unsupported task: {task}. task should be {Task.values()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
