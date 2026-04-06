# MDCNS: A Multi-Source Negative Sampling Framework for Sequential Recommendation

> Code repository for the SIGIR 2026 paper: **Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation**

## ✨ Overview

This repository contains the implementation of **MDCNS**, a sequential recommendation framework that combines **divergence** and **consensus** across multiple backbone recommenders for more effective negative sampling and training.

The repo is organized into two main parts:

- `MDCNS_Code/`: main implementation of MDCNS
- `Baselines/`: standalone implementations of baseline negative sampling methods

## 📁 Repository Structure

```text
SIGIR26-MDCNS/
├── MDCNS_Code/          # Main method, data, logs, and outputs
├── Baselines/           # Baseline implementations
├── SR_CR.pdf            # Paper / supplementary material
├── README.md            # Landing page
├── README.zh.md         # Chinese README
└── README.en.md         # English README
```

Key files and folders under `MDCNS_Code/`:

- `main.py`: main training entry
- `run_finetune.bash`: batch experiment launcher
- `data/`: processed data files and preprocessing scripts
- `Exp6_WeaktoStrong/`: training logs from batch runs
- `output/`: checkpoints and other outputs

Current baseline folders under `Baselines/`:

- `Neg_samples_DNS+`
- `Neg_samples_gnno`
- `Neg_samples_posmix`
- `Neg_samples_srns`
- `Neg_samples_two_pass`

## ⚙️ Environment

This repository does not currently provide a pinned `requirements.txt`. Based on the imported packages in the codebase, you will likely need:

- Python 3.9+
- PyTorch
- NumPy / SciPy / pandas / scikit-learn
- tqdm / texttable / openpyxl
- transformers
- `mamba-ssm`
- recbole

If you only run a subset of models, your effective dependency set may be smaller. Some backbones and baselines require additional packages such as `mamba-ssm` or `recbole`.

## 🚀 Quick Start

### 1. Enter the main code directory

```bash
cd MDCNS_Code
```

### 2. Prepare data

The main program reads data from `./data/` using the following naming convention:

```text
<Dataset>_train.txt
<Dataset>_val.txt
<Dataset>_test.txt
```

For example:

- `Beauty_train.txt`
- `Beauty_val.txt`
- `Beauty_test.txt`

The repository already includes processed splits for `Beauty`. Other datasets can be placed in `MDCNS_Code/data/` using the same naming pattern.

### 3. Run batch experiments

```bash
bash run_finetune.bash
```

Notes:

- The runnable script is `MDCNS_Code/run_finetune.bash`
- The script is configured for multi-GPU execution by default: `AVAILABLE_GPUS=(1 2 3 0)`
- You should adjust the GPU list, concurrency level, datasets, and hyperparameter ranges before launching experiments on your machine

## 🧪 Single Run Example

To run one experiment manually:

```bash
cd MDCNS_Code
python3 -u main.py \
  --data_name Beauty \
  --backbone SASRec \
  --backbone2 GRU4Rec \
  --hidden_size 64 \
  --loss_type BCE \
  --neg_sampler DNS \
  --N 100 \
  --K_hns 1 \
  --alpha 1.0 \
  --beta 1.0 \
  --d_lambda 3.5 \
  --dws_beta 1.0 \
  --temperature 0.0 \
  --kd_temperature 1.0 \
  --kd_gamma 0.01
```

## 📊 Outputs

Common output locations:

- Logs: `MDCNS_Code/Exp6_WeaktoStrong/`
- Checkpoints and training outputs: `MDCNS_Code/output/`

The batch script redirects stdout and stderr to per-run log files, which makes it easier to inspect results after large hyperparameter sweeps.

## 🗂️ Data and Preprocessing

Data-related files live in `MDCNS_Code/data/`, including:

- processed `Beauty` train/validation/test splits
- `data_process.py` and notebooks for preprocessing and analysis

The loading logic in `MDCNS_Code/main.py` expects:

```text
./data/<Dataset>_train.txt
./data/<Dataset>_val.txt
./data/<Dataset>_test.txt
```

## 🧩 Baselines

Each subdirectory in `Baselines/` is a relatively self-contained implementation and usually includes:

- `run_finetune.bash`
- `run_finetune_full.py`
- `models.py`
- `modules.py`
- `trainers.py`

To reproduce a specific baseline, enter its directory and run its local script independently.

## 📄 Paper Material

- Paper / material file: `SR_CR.pdf`

If you later add the camera-ready paper link, BibTeX, or benchmark tables, the root directory is the best place to keep those assets and index them here.
