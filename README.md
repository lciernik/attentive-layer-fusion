# Beyond the Final Layer: Attentive Multi-Layer Fusion for Vision Transformers

<div align="center">
  <img src="data/figure_1.png" alt="Overview of Attentive Multi-Layer Fusion for Vision Transformers" width="400">
</div>

This repository contains the code for the preprint paper "Beyond the Final Layer: Attentive Multi-Layer Fusion for Vision Transformers" [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Table of Contents

- [Repository Structure](#repository-structure)
- [Environment Setup and Setup of the Project](#environment-setup-and-setup-of-the-project)
- [Downloading the Datasets from Hugging Face](#downloading-the-datasets-from-hugging-face)
- [Running Experiments and Reproducing the Plots](#running-experiments-and-reproducing-the-plots)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Repository Structure

```
├── src/                   # Core source code
│   ├── data/              # Dataset handling and loading
│   ├── models/            # ThingsVision wrappers for feature extraction
│   ├── tasks/             # Experiment tasks (linear and attentive probes, hyperparameter tuning)
│   ├── eval/              # Evaluation metrics
│   └── utils/             # Utility functions
├── scripts/               # Experiment scripts and configurations
│   ├── configs/           # Configuration files for models and datasets
│   └── download_ds/       # Dataset download utilities
├── notebooks/             # Analysis and visualization notebooks
│   ├── main_section/     # Main paper figures
│   └── appendix_section/ # Appendix materials
└── data/                  # Static data files
```

**Note**: All experimental data (datasets, features, model checkpoints, results, etc.) is stored in the **project root directory** as described in [2. Project Structure](#2-project-structure).

## Environment Setup and Setup of the Project

### 1. Environment Setup

> [!NOTE]
> The experiments of this paper have been run on a SLURM compute cluster with Apptainer. Therefore, the following instructions are for running the experiments on this setup. 

We provide a docker image as well as an Apptainer container that contains all the necessary dependencies to rerun the experiments (on a SLURM).
- **Docker image**: `docker://ghcr.io/lciernik/attentive-layer-fusion:latest`
- **Apptainer container**: `oras://ghcr.io/lciernik/attentive-layer-fusion:latest-sif` 

### 2. Project Structure

All the project data (datasets, features, model checkpoints, etc.) is stored in the **project root directory**, defined in `scripts/project_location.py`. Please change it to your own project root directory and add the path to the container/image.

```
[PROJECT_ROOT]/
├── datasets/                     # Downloaded datasets
├── features/                     # Extracted features
├── models/                       # Trained probe models
├── model_similarities/           # Model similarity matrices
└── results/                      # Experimental results
```

You can build the project structure by running the script `scripts/project_location.py`.

## Downloading the Datasets from Hugging Face

All the datasets used in the experiments have been downloaded from the [CLIP Benchmark](https://huggingface.co/datasets/clip-benchmark) repository on Hugging Face. The script `scripts/download_ds/download_datasets.sh` downloads the datasets.

Ensure to use `[BASE_PATH_PROJECT]/datasets` as the target directory. Please follow the instructions in the [README](scripts/download_ds/README.md) to download the datasets.

## Running Experiments and Reproducing the Plots

> [!TIP]
> If you only want to reproduce the visualizations of the experiments, use the pre-aggregated results in <code>data/results/aggregated/</code> and run the notebooks in <code>notebooks/main_section</code>.

### Before You Start (on a SLURM Cluster with Apptainer)

- Connect to a compute node, via `srun`: `srun --partition=cpu-5h --mem=64G --pty bash`
- Run an interactive shell in the container: `apptainer run --nv --writable-tmpfs -B [BASE_PATH_PROJECT] [CONTAINER_PATH] /bin/bash`
- Navigate to the repository root directory: `cd $PATH_TO_REPO`
- Run any task (below)

### Running Different Tasks

To run any task, you must be in the container and in the repository root directory, i.e., at `PATH_TO_REPO`.

- **Feature extraction**:
  - General (entire datasets): `python scripts/feature_extraction.py`
  - ImageNet subset feature extraction:
    1. Create subset indices: `python scripts/generate_imagenet_subset_indices.py`
    2. Extract features: `python scripts/in_subset_extraction.py`
  - Create subset indices:
    - Run notebook `notebooks/check_wds_sizes_n_get_subsets.ipynb` to check the sizes of the datasets and get the subset indices.
- **Single model evaluation**: `python scripts/single_model_evaluation.py`
- **Combined representations evaluation**:
  - Linear probe: `python scripts/combined_models_evaluation_linear_probe.py`
  - Attentive probe: `python scripts/combined_models_evaluation_attn_probe.py`

> [!NOTE]
> One can also run the experiments on a **single machine** by using the commands stored as strings in the variable <code>job_cmd</code> in the abovementioned scripts.

### Visualizations

Before you can start reproducing the visualizations, you need to have the results of the experiments and aggregate them.

- **Aggregating the results**:
  - Run notebook `notebooks/aggregate_results.ipynb` to aggregate the results.
  - Or use the pre-aggregated results in <code>data/results/aggregated/</code>.
- **Visualizing the results**:
  - Run the notebooks in `notebooks/main_section` to visualize the results.

## Acknowledgments

TODO

## Citation

TODO
