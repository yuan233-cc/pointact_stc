#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DATASET="${1:-${SOURCE_DATASET:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/robot_data/franka}"
REPO_ID="${REPO_ID:-small_glass_uncap_pointact}"
CALIBRATION="${2:-${CALIBRATION:-${REPO_ROOT}/experiments/franka/calibration.json}}"
OUTPUT_DATASET="${OUTPUT_ROOT}/${REPO_ID}"
STATS_FILE="${OUTPUT_DATASET}/robot_state_action_stats/rot6d_points_d455.json"

if [[ -z "${SOURCE_DATASET}" ]]; then
  echo "Usage: $0 /path/to/lerobot_dataset [/path/to/calibration.json]" >&2
  echo "Alternatively set SOURCE_DATASET and CALIBRATION environment variables." >&2
  exit 2
fi

cd "${REPO_ROOT}"
python data_prep/franka_rgbd_to_pointact.py \
  --source "${SOURCE_DATASET}" \
  --output "${OUTPUT_DATASET}" \
  --repo-id "${REPO_ID}" \
  --calibration "${CALIBRATION}"

python data_prep/prepare_robot_state_action_stats.py \
  --dataset_dirs "${OUTPUT_DATASET}" \
  --output_file "${STATS_FILE}" \
  --point_cloud_dir points_d455 \
  --state_xyz_slice 0 3 \
  --action_xyz_slice 0 3 \
  --state_rotation_slice 3 7 \
  --action_rotation_slice 3 7 \
  --rotation_type quat \
  --target_rotation_type rot6d \
  --replace_zero_std

echo "Prepared ${OUTPUT_DATASET}"
echo "Training manifest: ${OUTPUT_DATASET}/franka_rgbd_manifest.json"
