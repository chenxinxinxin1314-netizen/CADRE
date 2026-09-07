# CADRE

Official PyTorch implementation of **CADRE: Stabilizing Feature Geometry for Distribution-Aware Classifier Recalibration in Exemplar-Free Class-Incremental Learning** for exemplar-free class-incremental learning.

Visual recognition systems must learn new categories without retraining on complete data histories, while privacy, storage, or access constraints may preclude retaining earlier images. Exemplar-free class-incremental learning addresses this setting, yet supervision limited to new-class data causes old representations to drift and classifier decisions to favor recent classes. CADRE addresses these failures at their respective stages. During representation learning, response losses for previous tasks are weighted by current teacher--student disagreement, while instance alignment and relational constraints preserve feature consistency and pairwise geometry. After the encoder is frozen, each class is described by a mixture of Gaussian sub-prototypes, from which pseudo-features are sampled to recalibrate the joint classifier. The preserved geometry keeps statistics estimated at earlier stages relevant to this correction. Only model parameters and compact class-conditional statistics are carried forward, with neither images nor individual historical features retained. Across three image-classification benchmarks and two task granularities, CADRE improves average incremental accuracy over the strongest competing method in each setting by 5.1--21.0 percentage points and obtains the highest final accuracy and lowest forgetting in all six settings. 

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


## Acknowledgment

This implementation is built on the FACIL class-incremental learning framework.
The original FACIL copyright and MIT license are retained in `LICENSE`.

## License

This repository is released under the MIT License. See `LICENSE` for details.
