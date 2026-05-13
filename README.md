# DFJSP-W Datasets

This repository provides training and validation datasets for the paper:

**Multi-Attention Offline Reinforcement Learning for Dual-Resource Job Shop Scheduling**

## Overview

DFJSP-W denotes a dual-resource flexible job shop scheduling setting with workers. Each operation must be assigned to a compatible machine and a compatible worker. The datasets are intended for offline reinforcement learning and scheduling research.

## Repository Structure

```text
DFJSP-W-Datasets/
|-- DataGenerator.py
|-- README.md
|-- train/
|   |-- 10x5x3+mix/
|   |-- 15x10x5+mix/
|   `-- 20x10x5+mix/
`-- valid/
    |-- 10x5x3+mix/
    |-- 15x10x5+mix/
    |-- 20x10x5+mix/
    |-- 30x10x8+mix/
    |-- 40x10x8+mix/
    |-- 50x10x8+mix/
    |-- 70x10x8+mix/
    |-- 100x10x8+mix/
    |-- 150x10x8+mix/
    |-- 200x10x8+mix/
    |-- 500x10x8+mix/
    `-- 1000x10x8+mix/
```

## Scale Notation

Each dataset directory follows the format:

```text
JxMxW+mix
```

where `J` is the number of jobs, `M` is the number of machines, and `W` is the number of workers. For example, `50x10x8+mix` contains instances with 50 jobs, 10 machines, and 8 workers.

## Data Format

Each instance is stored as a `.pkl` file containing:

- `job_lengths`: array of operation counts for each job, with shape `[n_jobs]`
- `op_pt`: operation-machine processing-time matrix, with shape `[total_ops, n_machines]`
- `op_wt`: operation-machine-worker processing-time matrix, with shape `[total_ops, n_machines, n_workers]`
- `instance_info`: basic metadata, including the number of jobs, machines, workers, and total operations

In `op_pt` and `op_wt`, a value of `0` means the corresponding machine or worker assignment is incompatible.

## Requirements

```bash
pip install numpy
```

## Loading Data

```python
from DataGenerator import load_single_instance

job_lengths, op_pt, op_wt, instance_info = load_single_instance(
    "valid/50x10x8+mix/50x10x8_001.pkl"
)
```

## Generating New Instances

`DataGenerator.py` can generate additional datasets with the same data format.

```bash
python DataGenerator.py \
  --n_jobs 50 \
  --n_machines 10 \
  --n_workers 8 \
  --num_instances 100 \
  --output_dir valid/50x10x8+mix \
  --dataset_name 50x10x8
```

The generator writes one `.pkl` file per instance and an optional JSON index file for batch loading.

## Citation

If you use this dataset in your research, please cite the corresponding paper:

```bibtex
@article{dfjspw_multi_attention_offline_rl,
  title  = {Multi-Attention Offline Reinforcement Learning for Dual-Resource Job Shop Scheduling},
  author = {To be updated},
  journal = {To be updated},
  year   = {To be updated}
}
```

The paper link and formal citation will be updated after publication.