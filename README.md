# Hierarchical Knowledge Graph Constraints in ECG Foundation Models: Post-Hoc Correction vs. Fine-Tuning

This repository is a cloned and extended version of [ECG-FM](https://github.com/bowang-lab/ECG-FM). Adjustments have been made for my BSc thesis, which explores whether incorporating a knowledge graph (KG) into the fine-tuning process improves hierarchical consistency of the ECG foundation model.

ECG diagnostic labels are not independent, some labels are hierarchically related in subtype relations. ECG foundation models do not seem to pick up on this from training on just ECG signal data, and can make hierarchical inconsistent predictions as a result. This project adds a KG-based penalty to the fine-tuning loss to encourage the model to respect these relationships, and compares this against baseline experiments with BCE-only and with post-hoc KG correction.

## Additions to ECG-FM

- **`kg_list.ipynb`** - builds the knowledge graph edge list (parent-child label relations) used for the penalty loss
- **`data_processing.ipynb`** - preprocesses MIMIC-IV-ECG data
- **`binary_cross_entropy_with_logits.py`** - modified version of fairseq-signals' BCE with logits-criterion to add the KG-penalty and some metric tracking
- **`graphs/`** - visualisations of the knowledge graph structure and training/results plots
- **`labeler.ipynb`** - builds the label set used for fine-tuning, based on ideas described in the ECG-FM paper

## Note on experiments

Actual model training and evaluation were run on the Snellius supercomputer. The SLURM job scripts and outputs are not in this repository, since they are specific to the Snellius setup.

## Setup

This project uses a conda environment (Python 3.10) with PyTorch and [fairseq-signals](https://github.com/Jwoo5/fairseq-signals).
