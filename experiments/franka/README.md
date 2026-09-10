# Franka D455 RGB-D training and deployment

This pipeline uses one visual camera (`observation.images.d455`) plus a point cloud built from
`observation.images.d455_depth`. A wrist camera is not used. Offline conversion and online
inference both call `pointact.utils.franka_rgbd.rgbd_to_point_cloud`, so depth scaling,
workspace filtering, voxelization, and point selection stay identical.
The default samples every second depth pixel before 1 cm voxelization; RGB remains 1280x720,
and both offline conversion and online inference use the same sampling setting.

## 1. Verify the fixed-D455 calibration

`calibration.json` is populated from the accepted `d455_result.yaml` in the
`eye_hand_calibration` ROS package.
It maps `d455_color_optical_frame` points into `base`, and its color intrinsics match the
1280x720 `aligned_depth_to_color` stream. Recalibrate and update this JSON if the camera mount
or resolution changes. `calibration.example.json` remains as a blank template:

- `intrinsics` is the 3x3 `K` matrix from the D455 **color** `CameraInfo`. The recorded depth
  topic is `aligned_depth_to_color`, so color intrinsics are required.
- `camera_to_base` is a 4x4 homogeneous transform mapping points from the D455 color optical
  frame into the Franka base frame used by `current_pose` and `target_pose`.

Do not use an identity transform unless the camera optical frame really is the robot base
frame. The converter validates matrix dimensions and the inference server verifies a SHA-256
fingerprint against the training manifest.

## 2. Convert the recorder dataset

Activate the PointAct environment and run:

```bash
conda activate pointact
cd /path/to/robot-PointAct
pip install -e .
bash experiments/franka/prepare_data.sh /path/to/lerobot_dataset
```

The command reads the original LeRobot-v3 dataset, writes a LeRobot-v2.1 copy compatible with
PointAct's pinned LeRobot version, creates `points_d455` LMDB entries keyed by
`episode_index-frame_index`, and computes centered rot6d action statistics. It deliberately
does not modify the source dataset. RGB-D preprocessing and temporary RGB writes each use four
bounded worker threads by default while retaining the original frame order; use the converter's
`--workers` and `--image-writer-threads` options when calling it directly to tune another machine.

## 3. Train

Download the Concerto checkpoint described in `INSTALLATION.md`, then run:

```bash
NUM_PROCESSES=1 PER_DEVICE_BATCH_SIZE=4 EPOCHS=100 \
  bash experiments/franka/train_pointact.sh \
    /path/to/concerto_large.pth \
    /path/to/processed_dataset
```

The second argument can point to any completed dataset produced by `prepare_data.sh`, on any
mounted disk. The training script validates it and generates a temporary PointAct data YAML.
Alternatively set `PROCESSED_DATASET=/path/to/processed_dataset` or provide a custom YAML with
`DATA_CONFIG=/path/to/data.yaml`. If the second argument is omitted, the repository-local
`robot_data/franka/small_glass_uncap_pointact` default is used.

The supplied config trains a 40-step absolute EEF action chunk at the dataset's 10 Hz rate.
It uses D455 RGB for the VLM and D455 RGB-D geometry for PointAct. Robot state and wrist images
are intentionally disabled, matching deployment.

## 4. Start the websocket policy server

```bash
python scripts/serve_franka_ws.py \
  --checkpoint /path/to/checkpoint-final-N \
  --calibration experiments/franka/calibration.json \
  --training-manifest robot_data/franka/small_glass_uncap_pointact/franka_rgbd_manifest.json \
  --host 0.0.0.0 --port 8765
```

The server accepts the existing `robot_ws_client` JSON protocol and returns `action_chunk`
messages containing absolute `[x,y,z,qx,qy,qz,qw]` targets and normalized gripper commands.
On an inference exception it returns a one-step hold command by default.

## 5. Connect the ROS client safely

Rebuild the modified client and test without commanding the robot:

```bash
cd /path/to/ros_ml_ws
colcon build --packages-select robot_ws_client --symlink-install
source install/local_setup.bash
ros2 launch robot_ws_client pointact_ws_client.launch.py \
  server_uri:=ws://SERVER_IP:8765 dry_run:=true
```

Confirm the server reports a small RGB/depth timestamp skew, inspect
`/pointact_ws_client/debug_target_pose`, and validate workspace/action bounds before changing
`dry_run:=false`. Keep `command_hz:=10.0` to match the training data.
