#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_CONFIG="${DATA_CONFIG:-${REPO_ROOT}/experiments/franka/data-franka-d455-point.yaml}"
PROCESSED_DATASET="${2:-${PROCESSED_DATASET:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/franka-small-glass-pointact}"
VLM="${VLM:-Qwen/Qwen2.5-VL-3B-Instruct}"
PTV3_CKPT="${1:-${PTV3_CKPT:-}}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-100}"

if [[ -z "${PTV3_CKPT}" ]]; then
  echo "Usage: $0 /path/to/concerto_large.pth [/path/to/processed_dataset]" >&2
  echo "Alternatively set the PTV3_CKPT environment variable." >&2
  exit 2
fi

GENERATED_DATA_CONFIG=""
if [[ -n "${PROCESSED_DATASET}" ]]; then
  GENERATED_DATA_CONFIG="$(mktemp "${TMPDIR:-/tmp}/pointact-franka-data.XXXXXX.yaml")"
  trap 'rm -f "${GENERATED_DATA_CONFIG}"' EXIT
  python "${REPO_ROOT}/experiments/franka/make_data_config.py" \
    --dataset "${PROCESSED_DATASET}" \
    --output "${GENERATED_DATA_CONFIG}"
  DATA_CONFIG="${GENERATED_DATA_CONFIG}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export DATASET_NUM_PROCESSES="${DATASET_NUM_PROCESSES:-1}"

accelerate launch --num_processes "${NUM_PROCESSES}" scripts/train.py \
  --model_class VLAEncDec3DWithActionRegressionModel \
  --output_dir "${OUTPUT_DIR}" \
  --vlm-name-or-path "${VLM}" \
  --data-path "${DATA_CONFIG}" \
  --chunk-size 40 \
  --dataloader-num-workers 4 \
  --freeze-vision-tower True \
  --freeze-llm True \
  --freeze-merger True \
  --bf16 True \
  --tf32 True \
  --num-train-epochs "${EPOCHS}" \
  --per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --merger-lr 5e-5 \
  --vision-lr 2e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --gradient-checkpointing True \
  --save-strategy steps \
  --logging-steps 10 \
  --save-steps 500 \
  --save-total-limit 3 \
  --report-to tensorboard \
  --image-aug True \
  --color-aug True \
  --use-robot-state False \
  --ctx-embed-size 512 \
  --ptv3-backend concerto \
  --ptv3-patch-size 1024 \
  --ptv3-enc-mode True \
  --ptv3-enc-channels 64 128 256 512 768 \
  --ptv3-enc-depths 3 3 3 12 3 \
  --ptv3-enc-num-head 4 8 16 32 48 \
  --ptv3-input-channels 6 \
  --ptv3-apply-point-ca False \
  --ptv3-init-ckpt-file "${PTV3_CKPT}" \
  --action-regression-loss l2 \
  --regression-head-heatmap-temp 0.1 \
  --action-head-pos-center zero \
  --max-state-dim 1 \
  --max-action-dim 10
