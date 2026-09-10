#!/usr/bin/env python3
"""Serve a PointAct checkpoint to robot_ws_client over its JSON WebSocket protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.run_server import Policy, ServerArgs
except ModuleNotFoundError:  # Direct execution puts scripts/ rather than the repo root on sys.path.
    from run_server import Policy, ServerArgs
from pointact.utils.franka_rgbd import (
    FrankaRGBDCalibration,
    decode_ros_depth,
    decode_ros_rgb,
    rgbd_to_point_cloud,
    validated_target_pose,
)

LOGGER = logging.getLogger("pointact.franka_ws")
RGB_KEY = "observation.images.d455"
DEPTH_KEY = "observation.images.d455_depth"


def normalized_gripper(obs: dict[str, Any], max_finger_width: float) -> float:
    width = obs.get("gripper_width")
    return 0.0 if width is None else float(np.clip(float(width) / max_finger_width, 0.0, 1.0))


class FrankaPointActPolicy:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.calibration = FrankaRGBDCalibration.load(args.calibration)
        self.policy = Policy(
            ServerArgs(
                pretrained_path=str(args.checkpoint),
                num_denoise_steps=args.num_denoise_steps,
            )
        )
        robot_config = self.policy.processor.robot_config
        repo_ids = list(robot_config.get("select_action_keys", {}))
        if not repo_ids:
            raise ValueError("checkpoint processor has no PointAct robot_config")
        self.repo_id = args.repo_id or repo_ids[0]
        if self.repo_id not in repo_ids:
            raise ValueError(f"repo_id {self.repo_id!r} not in checkpoint: {repo_ids}")
        video_keys = robot_config["select_video_keys_for_vlm"][self.repo_id]
        if video_keys != [RGB_KEY]:
            raise ValueError(f"checkpoint must use only {RGB_KEY!r}, got {video_keys}")
        with args.training_manifest.open(encoding="utf-8") as f:
            manifest = json.load(f)
        expected = manifest.get("calibration_sha256")
        if not expected or expected != self.calibration.fingerprint():
            raise ValueError("deployment calibration differs from the calibration used for training")

    def infer(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        rgb_payload = obs.get(RGB_KEY)
        depth_payload = obs.get(DEPTH_KEY)
        if not isinstance(rgb_payload, dict) or not isinstance(depth_payload, dict):
            raise ValueError(f"observation must contain {RGB_KEY!r} and {DEPTH_KEY!r}")
        skew = abs(float(rgb_payload.get("stamp", 0.0)) - float(depth_payload.get("stamp", 0.0)))
        if skew > self.args.max_rgb_depth_skew_ms / 1000.0:
            raise ValueError(f"D455 RGB/depth timestamps differ by {skew * 1000:.1f} ms")
        rgb = decode_ros_rgb(rgb_payload)
        depth = decode_ros_depth(depth_payload)
        points = rgbd_to_point_cloud(rgb, depth, self.calibration)
        prompt = str(obs.get("prompt") or self.args.default_prompt)
        batch = {
            RGB_KEY: np.expand_dims(rgb, axis=0),
            "observation.points": [points],
            "observation.points.voxelized": [True],
            "task": [prompt],
            "repo_id": [self.repo_id],
        }
        result = self.policy.get_action(batch, {"pred_rot_type": self.args.pred_rot_type})
        actions = np.asarray(result["action"])
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] < 8:
            raise ValueError(f"expected PointAct actions [horizon, >=8], got {actions.shape}")
        return [
            {
                "target_pose": validated_target_pose(action, self.calibration.workspace),
                "gripper": float(np.clip(action[7], 0.0, 1.0)),
            }
            for action in actions
        ]

    def hold(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        return [{
            "target_pose": [float(value) for value in obs["ee_pose"]],
            "gripper": normalized_gripper(obs, self.args.gripper_max_finger_width),
        }]


async def serve(args: argparse.Namespace) -> None:
    try:
        from websockets.asyncio.server import serve as websocket_serve
    except ImportError as exc:
        raise ImportError("install the updated PointAct dependencies with: pip install -e .") from exc
    policy = FrankaPointActPolicy(args)

    async def handler(websocket) -> None:
        LOGGER.info("connection opened from %s", websocket.remote_address)
        async for raw in websocket:
            obs = None
            try:
                obs = json.loads(raw)
                if obs.get("type") != "observation":
                    await websocket.send(json.dumps({"type": "heartbeat"}))
                    continue
                actions = policy.infer(obs)
            except Exception:
                LOGGER.exception("inference failed")
                if not args.hold_on_error or obs is None or "ee_pose" not in obs:
                    raise
                actions = policy.hold(obs)
            await websocket.send(json.dumps({
                "type": "action_chunk",
                "seq": obs.get("seq"),
                "actions": actions,
            }))
            LOGGER.info("seq=%s returned %d actions", obs.get("seq"), len(actions))

    async with websocket_serve(handler, args.host, args.port, max_size=None):
        LOGGER.info("PointAct Franka server listening on ws://%s:%d", args.host, args.port)
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--repo-id", default=None, help="defaults to the checkpoint's first dataset")
    parser.add_argument("--default-prompt", default="pick up the small glass jars and put them into the container")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pred-rot-type", choices=("quat", "euler", "rot6d"), default="rot6d")
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--max-rgb-depth-skew-ms", type=float, default=50.0)
    parser.add_argument("--gripper-max-finger-width", type=float, default=0.04)
    parser.add_argument("--no-hold-on-error", dest="hold_on_error", action="store_false")
    parser.set_defaults(hold_on_error=True)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(parse_args()))
