# DFJSP-W Datasets

This repository provides the **training and validation datasets** used in the paper:

**Multi-Attention Offline Reinforcement Learning for Dual-Resource Job Shop Scheduling**

---

## Repository Structure

The dataset is organized by problem scale and split into training and validation sets.

```text
DFJSP-W-Datasets/
├── train/
│   ├── 10x5x3/
│   ├── 15x10x5/
│   ├── 20x10x5/
├── valid/
│   ├── 10x5x3/
│   ├── 15x10x5/
│   ├── 20x10x5/
│   ├── 30x10x8/
│   └── 40x10x8/

---

## Dataset Scale Notation
Each instance scale is denoted as:n × m × w

## Usage
The datasets are intended for offline learning and scheduling research.
Training and validation sets are separated as used in the experiments.
Users are free to load and preprocess the data according to their own implementations.

## Citation
If you use this dataset in your research, please cite the corresponding paper:

