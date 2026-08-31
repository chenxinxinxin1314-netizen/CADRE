# CADRE

Official PyTorch implementation of **CADRE: Calibrated Adaptive Distillation
with Relational Embedding** for exemplar-free class-incremental learning.

CADRE addresses representation drift and classifier bias in two coupled
phases. During incremental representation learning, adaptive response
distillation assigns task-dependent weights to previous-task responses, while
pointwise and relational feature constraints preserve the embedding space.
After each task, class distributions are summarized with Gaussian
sub-prototypes and the classifier heads are recalibrated in feature space with
the backbone frozen. No image or individual historical feature is retained.

## Repository Structure

```text
src/approach/cadre.py             CADRE implementation
src/approach/incremental_learning.py
src/datasets/                     Dataset loading and task construction
src/networks/                     Backbone and incremental network definitions
src/loggers/                      Experiment logging
src/main_incremental.py           Training entry point
scripts/                          Reproduction scripts
```

Only the full CADRE method is exposed. The public implementation does not
include the internal ablation switches used during method development.

## Installation

Create the tested environment with Conda:

```bash
conda env create -f environment.yml
conda activate cadre
```

Alternatively, install the Python dependencies directly:

```bash
pip install -r requirements.txt
```

The current release was tested with Python 3.10, PyTorch 2.11, and torchvision
0.26. Install a PyTorch build compatible with the CUDA version on your system.

## Data Preparation

By default, datasets are read from `data/` in the repository root. A different
location can be selected without editing source code:

```bash
export CADRE_DATA_ROOT=/path/to/datasets
```

Expected dataset directories are:

```text
data/
|-- cifar100/                  downloaded automatically by torchvision
|-- tiny-imagenet-200/
`-- imagenet_subset/
    |-- train/<class_name>/*.JPEG
    `-- val/<class_name>/*.JPEG
```

Tiny-ImageNet must retain its standard extracted directory structure. For
Tiny-ImageNet and ImageNet-Subset, the loader creates the required split index
files on first use.

## Training

Run CIFAR-100 with 10 incremental tasks:

```bash
GPU=0 SEED=2 TASKS=10 bash scripts/run_cifar100.sh
```

Run Tiny-ImageNet or ImageNet-Subset in the same way:

```bash
GPU=0 SEED=2 TASKS=10 bash scripts/run_tiny_imagenet.sh
GPU=0 SEED=2 TASKS=10 bash scripts/run_imagenet_subset.sh
```

Set `TASKS=20` for the 20-task protocol. Results are written under
`results/<dataset>/<TASKS>task/seed<SEED>/` unless `RESULTS_DIR` is provided.

The entry point can also be invoked directly:

```bash
python src/main_incremental.py \
  --approach cadre \
  --datasets cifar100 \
  --network resnet32 \
  --num-tasks 10 \
  --seed 2 \
  --gpu 0
```

## Default CADRE Configuration

| Parameter | Value |
| --- | ---: |
| Response-distillation coefficient | 3.0 |
| Distillation temperature | 1.0 |
| Feature-distillation coefficient | 1.0 |
| Adaptive weighting temperature | 2.0 |
| Relational coefficient | 1.0 |
| Maximum sub-prototypes per class | 3 |
| Classifier calibration epochs | 10 |

## Citation

The citation entry will be updated when the paper is published. For the current
manuscript, please use:

```bibtex
@article{chen2026cadre,
  title   = {CADRE: Calibrated Adaptive Distillation with Relational Embedding},
  author  = {Chen, Xin and Wang, Zhuowei},
  year    = {2026},
  note    = {Manuscript}
}
```

## Acknowledgment

This implementation is built on the FACIL class-incremental learning framework.
The original FACIL copyright and MIT license are retained in `LICENSE`.

## License

This repository is released under the MIT License. See `LICENSE` for details.
