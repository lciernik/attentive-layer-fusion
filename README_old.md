# Rep2Rep

## Setting up environment on the cluster

To be able to run Slurm and Apptainer from within Apptainer, you first have to add the following line to your `.zshrc`/`.bashrc` file on the cluster (**NOTE: change `PATH_TO_REPO`** to the location of the cloned repo on the cluster):

```bash
export PATH_TO_REPO=/home/[PATH TO THE CLONED REPO]/rep2rep
```

To run any stript or simply get into the container, run the following commands.

1. Connect to a compute node, via `srun` (more info [here](https://wiki.tuflow.com/index.php/Running_Jobs)):

   - CPU: `srun  --partition=cpu-5h --mem=64G --pty bash`
   - GPU: `srun  --gpus=1 --partition=gpu-2h --mem=64G --pty bash`

2. For the **first time**, you have to login to DockerHub (i.e., store the DockerHub credentials):

   ```bash
   apptainer remote login --username <github name> docker://ghcr.io
   ```
   - **NOTE**: You have to have a personal access token for GitHub. See [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic) how to create one.
3. Also for the **first time** on the cluster, create a new SSH key pair in case you don't have one yet
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
   and add your public key to the `authorized_keys`:

   ```bash
   cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
   ```

   You can verify that this works by running

   ```bash
   ssh $USER@$HOST exit
   ```
   which should return without any prompt.
3. To run a script, e.g., `scripts/single_model_evaluation.py`, you have to do the following steps:
   a. On the cluster connect to a compute node, see 1. above.
   b. Run an interactive shell in the container:
      ```bash
      apptainer run --nv --writable-tmpfs -B /home/space/rep2rep oras://ghcr.io/lciernik/rep2rep:latest-sif /bin/bash
      ```
   c. Navigate to the repository root directory and run the script:
      ```bash
      cd $PATH_TO_REPO
      python scripts/single_model_evaluation.py
      ```
   d. Exit the container.

In general, to run an interactive shell in the container, you can use the following command:

```bash
apptainer run --nv --writable-tmpfs -B /home/space/rep2rep oras://ghcr.io/lciernik/rep2rep:latest-sif /bin/bash
```

To run a jupyter notebook, you can use the following command:

```bash
apptainer run --nv --writable-tmpfs -B /home/space/rep2rep oras://ghcr.io/lciernik/rep2rep:latest-sif jupyter notebook --ip 0.0.0.0 --no-browser --port 8888
```


#### Installing new packages
If you want to install new packages, you can do so by installing them tempoarily in the container, via 
`uv add <package_name>`. If you want to install them permanently, you have to tag and push the repository. This will trigger the GitHub Action workflow to build the new image and push it to the DockerHub.

```
git tag v*.*.*
git push origin v*.*.*
```
This will trigger the GitHub Action workflow to build the new image and push it to the DockerHub.


## WandB usage
- Environment variables in your `.zshrc`/`.bashrc` file: `export WANDB_API_KEY=<your_api_key>`
- To use WandB, you have to set the `use_wandb` flag to `True`, see argument `use_wandb` in the `argparse.py` file.
- See `utils/wandb_logger.py` for more details.

(TODO: add more details)


## Pre-requisites

1. **Define the project root directory in `scripts/project_location.py`**!
   - If you choose to use different paths than the default you might want to consider to create symlinks to the old project such that datasets don't have to be downloaded again and features precomputed.

## Table of Contents

1. [Repository and project overview](#repository-and-project-overview)
2. [How to install?](#how-to-install)
3. [How to run?](#how-to-run)

## Repository and project overview

### Repository organization

- 🔧 `sim_consistency/`
  - Core library code .
- 📜 `scripts/`
  - `config/`: Configuration files for models and datasets
  - `download_ds/`: Dataset download scripts
  - Scripts for feature extraction, model similarity computation, and linear probe evaluation. (
    See [How to Run](#how-to-run))
- 📓 `notebooks/`
  - Jupyter notebooks for analysis and visualization.
    Each notebook is named according to its corresponding paper section and can be used to reproduce our findings 🧪 .

### Project structure

The code relies on a specific directory structure for data organization.
All paths are configured and created in `scripts/project_location.py`.

```plaintext
project_root/
├── datasets/           # Raw and subsetted datasets
├── features/           # Extracted model features
├── model_similarities/ # Representation  similarity matrices for a given dataset, set of models, and similarity metric
├── models/             # Trained linear probe models
└── results/            # Evaluation results and experiments
```

## How to install?

1. Nagivate to the repository root directory.
2. Install the package: `pip install .`
3. Configure the project location as described in the [Project structure](#project-structure) section.
   You can define the paths in the `scripts/project_location.py` file and run it. It will create the necessary directories.

## How to run?

⚠️ **Disclaimer**: Scripts are designed for SLURM clusters and are computationally expensive. For local reproduction,
we recommend downloading our intermediate results [here](https://tubcloud.tu-berlin.de/s/iTmTwqnao3fxH2t) (after step **4.1**) and proceeding directly to [step 4.2](#4-how-to-reproduce-our-results).

### 0. Download datasets 💾

- Create the webdatasets directory `[PROJECT_ROOT]/datasets/wds`
- Download the datasets using the script `scripts/download_ds/download_webdatasets.sh`.

```bash
cd scripts/download_ds
bash download_webdatasets.sh [PROJECT_ROOT]/datasets/wds
```

**NOTE**: Scripts download datasets from `huggingface.co`. A Hugging Face account may be required -
see [download guide](https://huggingface.co/docs/hub/datasets-downloading#using-git).

### 1. Feature Extraction 🔍

Running the script `scripts/feature_extraction.py` will extract features from the models specified in the
`models_config` file for the datasets specified in the `datasets` file. The script launches a SLURM job for each model
separately. It saves the extracted features in `[PROJECT_ROOT]/features` directory.

```bash
cd scripts
python feature_extraction.py \
       --models_config ./configs/models_config_wo_alignment.json \
       --datasets ./configs/webdatasets_w_in1k.txt
```

### 2. Model Similarities 🔄

The computation of the model similarities is a crucial step in our analysis. It consists of two parts:
Dataset subsampling and model similarity computation. The first part is necessary to ensure that the datasets have a
maximum of 10k samples (see paper for justification), while the second part computes the representational similarities
between the models.

#### Dataset subsampling

- **ImageNet-1k**:
  1. Run:
     ```bash
        cd scripts
        python generate_imagenet_subset_indices.py
     ```
     It will generate the indices for the ImageNet-1k subset datasets and store them in `[PROJECT_ROOT]/datasets/subset`.
  2. Create the new ImageNet-1k subsets by slicing the precomputed features:
     ```bash
        cd scripts
        python in_subset_extraction.py
     ```
     It will create the ImageNet-1k subsets and store them in `[PROJECT_ROOT]/datasets/imagenet-subset-{X}k`
     (`X` indicates the total nr. of samples).
- **Remaining Datasets**: Run the jupyter notebook `notebooks/check_wds_sizes_n_get_subsets.ipynb` to check the dataset sizes and create
  subsets of the datasets if needed. The indices for the subsets are stored in `[PROJECT_ROOT]/datasets/subset`.

#### Model similarity computation

Running the script `scripts/distance_matrix_computation.py` will compute the representational similarities between the
models for each dataset and similarity metric specified in `scripts/configs/similarity_metric_config_local_global.json`.
It saves the computed similarity matrices in `[PROJECT_ROOT]/model_similarities` directory.

```bash
cd scripts
python distance_matrix_computation.py \
       --models_config ./configs/models_config_wo_alignment.json \
       --datasets ./configs/webdatasets_w_insub10k.txt
```

### 3. Linear Probing (Single model downstream task evaluation) 📈

Running the script `scripts/single_model_evaluation.py` will train a linear probe on the extracted features for each
model and dataset specified in the `models_config` and `datasets` files, respectively.
It saves the trained models in `[PROJECT_ROOT]/models` and the evaluation results in the
`[PROJECT_ROOT]` directory.

```bash
cd scripts
python single_model_evaluation.py \
       --models_config ./configs/models_config_wo_alignment.json \
       --datasets ./configs/webdatasets_w_in1k.txt
```

**Note**: The script automatically launches separate SLURM jobs for each model to enable parallel processing.

### 4. How to reproduce our experiments? 🧪

After having extracted the features, computed the model similarities, and trained the linear probes, you can reproduce
our results by following steps:

1. Run aggregation notebooks:
   - All `notebooks/aggregate_*` notebooks: store the results in `[PROJECT_ROOT]/ results/aggregated/`
   - 🔥 For consistency computation of **specific model set pairs** 🔥:
     `notebooks/aggregate_consistencies_for_specific_model_set_pairs.ipynb`
2. Run section-specific notebooks to generate figures
   - Results saved in `results/plots/`
