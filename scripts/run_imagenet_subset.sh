#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
SEED="${SEED:-2}"
TASKS="${TASKS:-10}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/imagenet_subset/${TASKS}task/seed${SEED}}"
export CADRE_DATA_ROOT="${CADRE_DATA_ROOT:-${ROOT_DIR}/data}"

python -u "${ROOT_DIR}/src/main_incremental.py" \
  --gpu "${GPU}" \
  --results-path "${RESULTS_DIR}" \
  --datasets imagenet_subset \
  --num-tasks "${TASKS}" \
  --network resnet32 \
  --approach cadre \
  --seed "${SEED}" \
  --batch-size 64 \
  --num-workers 2 \
  --nepochs 300 \
  --lr 0.1 \
  --lr-factor 3 \
  --lr-patience 10 \
  --lr-min 0.0001 \
  --clipping 1 \
  --momentum 0.9 \
  --weight-decay 0 \
  --lambda-distill 3.0 \
  --distill-temperature 1.0 \
  --lambda-feat-distill 1.0 \
  --adaptive-temp 2.0 \
  --lambda-relation 1.0 \
  --num-subprototypes 3 \
  --min-samples-for-multimodal 10 \
  --covariance-epsilon 0.0001 \
  --balanced-bs 128 \
  --balanced-epochs 10 \
  --balanced-lr 0.01
