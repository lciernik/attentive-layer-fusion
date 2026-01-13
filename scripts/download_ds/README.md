# Dataset Download Script

This script automates the download of CLIP benchmark datasets from Hugging Face repositories.

## Overview

The `download_datasets.sh` script reads a text file containing dataset repository names and clones them from Hugging Face using Git LFS (Large File Storage) to handle large dataset files.

## Prerequisites

Before running the script, ensure you have:

1. **Git LFS installed**: The script uses Git LFS to handle large files
2. **SSH access to Hugging Face**: The script uses SSH URLs (`git@hf.co:datasets/clip-benchmark/...`)
3. **Proper SSH key setup**: Make sure your SSH key is configured for Hugging Face

### Setting up SSH access to Hugging Face

1. Generate an SSH key if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. Add your public key to your Hugging Face account:
   - Copy your public key: `cat ~/.ssh/id_ed25519.pub`
   - Go to [Hugging Face Settings](https://huggingface.co/settings/keys)
   - Add the SSH key

3. Test the connection:
   ```bash
   ssh -T git@hf.co
   ```

## Usage

### Basic Syntax

```bash
./download_datasets.sh <target_directory>
```

### Parameters

- `<target_directory>`: The directory where the datasets will be cloned (required) **NOTE: please ensure that the directory exists and it fits into the project structure described in the root README.md.**

### Example

```bash
# Navigate to the script directory
cd /path/to/scripts/download_ds

# Make the script executable
chmod +x download_datasets.sh

# Run the script
./download_datasets.sh /path/to/your/datasets/directory
```
