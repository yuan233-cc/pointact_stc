#!/usr/bin/env python3
"""Generate a PointAct data YAML for an arbitrary processed Franka dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    required = [
        dataset / "meta" / "info.json",
        dataset / "points_d455",
        dataset / "robot_state_action_stats" / "rot6d_points_d455.json",
        dataset / "franka_rgbd_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("processed dataset is incomplete; missing: " + ", ".join(missing))

    repo_id = args.repo_id or dataset.name
    config = {
        "mm_datasets": [],
        "lerobot_datasets": [{
            "repo_id": repo_id,
            # PointAct's MultiLeRobotDataset resolves the final data path as
            # Path(root) / repo_id, so root must be the dataset's parent.
            "root": str(dataset.parent),
            "class_name": "LeRobotPointCloudDataset",
            "select_video_keys": ["observation.images.d455"],
            "video_key_ids_for_vlm": [0],
            "select_state_keys": [],
            "select_action_keys": ["action"],
            "converted_rot_type": "rot6d",
            "is_delta_action": False,
            "is_action_eef": True,
            "points_workspace": None,
            "max_npoints": 4096,
            "augment_pc_rot": 0,
            "point_cloud_dirname": "points_d455",
            "state_action_norm_file": str(required[2]),
        }],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Wrote PointAct data config for {dataset} to {args.output}")


if __name__ == "__main__":
    main()
