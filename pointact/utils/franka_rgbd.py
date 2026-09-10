"""Shared D455 RGB-D preprocessing for Franka training and deployment."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class FrankaRGBDCalibration:
    """D455 color intrinsics and optical-camera-to-Franka-base transform."""

    width: int
    height: int
    intrinsics: np.ndarray
    camera_to_base: np.ndarray
    depth_scale_m: float = 0.001
    min_depth_m: float = 0.1
    max_depth_m: float = 2.0
    pixel_stride: int = 1
    voxel_size_m: float = 0.01
    max_points_before_training: int = 20000
    workspace: dict[str, list[float]] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "FrankaRGBDCalibration":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float64)
        camera_to_base = np.asarray(raw["camera_to_base"], dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError(f"{path}: intrinsics must be 3x3, got {intrinsics.shape}")
        if camera_to_base.shape != (4, 4):
            raise ValueError(f"{path}: camera_to_base must be 4x4, got {camera_to_base.shape}")
        if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(camera_to_base)):
            raise ValueError(f"{path}: calibration matrices must contain only finite values")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError(f"{path}: fx and fy must be positive")
        calibration = cls(
            width=int(raw["width"]),
            height=int(raw["height"]),
            intrinsics=intrinsics,
            camera_to_base=camera_to_base,
            depth_scale_m=float(raw.get("depth_scale_m", 0.001)),
            min_depth_m=float(raw.get("min_depth_m", 0.1)),
            max_depth_m=float(raw.get("max_depth_m", 2.0)),
            pixel_stride=int(raw.get("pixel_stride", 1)),
            voxel_size_m=float(raw.get("voxel_size_m", 0.01)),
            max_points_before_training=int(raw.get("max_points_before_training", 20000)),
            workspace=raw.get("workspace"),
        )
        calibration.validate()
        return calibration

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("calibration width and height must be positive")
        if self.depth_scale_m <= 0 or not 0 <= self.min_depth_m < self.max_depth_m:
            raise ValueError("invalid depth scale/range")
        if self.pixel_stride <= 0 or self.voxel_size_m <= 0:
            raise ValueError("pixel_stride and voxel_size_m must be positive")
        if self.max_points_before_training <= 0:
            raise ValueError("max_points_before_training must be positive")
        if self.workspace is not None:
            for key in ("X_BBOX", "Y_BBOX", "Z_BBOX"):
                bounds = self.workspace.get(key)
                if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] >= bounds[1]:
                    raise ValueError(f"workspace.{key} must be [min, max]")

    def fingerprint(self) -> str:
        payload = {
            field.name: (getattr(self, field.name).tolist() if isinstance(getattr(self, field.name), np.ndarray)
                         else getattr(self, field.name))
            for field in dataclasses.fields(self)
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validated_target_pose(
    action: np.ndarray,
    workspace: dict[str, list[float]] | None,
) -> list[float]:
    """Validate an absolute Franka target and normalize its xyzw quaternion."""
    pose = np.asarray(action[:7], dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"model returned an invalid target pose: {pose}")
    quaternion_norm = float(np.linalg.norm(pose[3:7]))
    if quaternion_norm < 1e-8:
        raise ValueError("model returned a zero quaternion")
    pose[3:7] /= quaternion_norm
    if workspace is not None:
        for axis, key in enumerate(("X_BBOX", "Y_BBOX", "Z_BBOX")):
            low, high = workspace[key]
            if not low <= pose[axis] <= high:
                raise ValueError(f"predicted {key[0].lower()}={pose[axis]:.3f} is outside [{low}, {high}]")
    return pose.tolist()


def unpack_recorded_depth(packed_rgb: np.ndarray) -> np.ndarray:
    """Decode franka_data_recorder's lossless depth PNG into uint16 millimetres."""
    packed = np.asarray(packed_rgb, dtype=np.uint8)
    if packed.ndim != 3 or packed.shape[2] != 3:
        raise ValueError(f"packed depth must be HxWx3, got {packed.shape}")
    depth = (packed[..., 0].astype(np.uint16) << 8) | packed[..., 1].astype(np.uint16)
    return np.where(packed[..., 2] != 0, depth, 0).astype(np.uint16)


def decode_ros_rgb(payload: dict[str, Any]) -> np.ndarray:
    """Decode the JSON representation emitted by robot_ws_client into RGB uint8."""
    encoding = str(payload["encoding"]).lower()
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"RGB payload requires rgb8 or bgr8, got {encoding!r}")
    height, width, step = int(payload["height"]), int(payload["width"]), int(payload["step"])
    rows = np.frombuffer(base64.b64decode(payload["data_b64"]), dtype=np.uint8).reshape(height, step)
    rgb = rows[:, : width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        rgb = rgb[..., ::-1]
    return np.ascontiguousarray(rgb)


def decode_ros_depth(payload: dict[str, Any]) -> np.ndarray:
    """Decode a 16UC1/mono16 JSON ROS image payload into native uint16 values."""
    encoding = str(payload["encoding"]).lower()
    if encoding not in ("16uc1", "mono16"):
        raise ValueError(f"depth payload requires 16UC1 or mono16, got {encoding!r}")
    height, width, step = int(payload["height"]), int(payload["width"]), int(payload["step"])
    if step < width * 2:
        raise ValueError(f"depth step {step} is smaller than packed row size {width * 2}")
    rows = np.frombuffer(base64.b64decode(payload["data_b64"]), dtype=np.uint8).reshape(height, step)
    pixels = np.ascontiguousarray(rows[:, : width * 2]).view(np.dtype("<u2")).reshape(height, width)
    message_bigendian = bool(int(payload.get("is_bigendian", 0)))
    if message_bigendian:
        pixels = pixels.byteswap()
    return pixels.astype(np.uint16, copy=False)


def rgbd_to_point_cloud(
    rgb: np.ndarray,
    depth_values: np.ndarray,
    calibration: FrankaRGBDCalibration,
) -> np.ndarray:
    """Create deterministic xyzrgb points in the Franka base frame.

    This exact function is used by offline conversion and the websocket server.
    RGB must already be aligned with depth (the recorder uses aligned_depth_to_color).
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth_values = np.asarray(depth_values)
    expected = (calibration.height, calibration.width)
    if rgb.shape != (*expected, 3) or depth_values.shape != expected:
        raise ValueError(f"RGB/depth must be {expected}; got {rgb.shape} and {depth_values.shape}")

    stride = calibration.pixel_stride
    depth_m = depth_values[::stride, ::stride].astype(np.float64) * calibration.depth_scale_m
    colors = rgb[::stride, ::stride].astype(np.float32) / 255.0
    vv, uu = np.mgrid[0 : calibration.height : stride, 0 : calibration.width : stride]
    valid = np.isfinite(depth_m) & (depth_m >= calibration.min_depth_m) & (depth_m <= calibration.max_depth_m)
    fx, fy = calibration.intrinsics[0, 0], calibration.intrinsics[1, 1]
    cx, cy = calibration.intrinsics[0, 2], calibration.intrinsics[1, 2]
    z = depth_m[valid]
    camera_xyz = np.column_stack(((uu[valid] - cx) * z / fx, (vv[valid] - cy) * z / fy, z))
    base_xyz = camera_xyz @ calibration.camera_to_base[:3, :3].T + calibration.camera_to_base[:3, 3]
    points = np.concatenate((base_xyz.astype(np.float32), colors[valid]), axis=1)

    if calibration.workspace is not None:
        w = calibration.workspace
        keep = (
            (points[:, 0] > w["X_BBOX"][0]) & (points[:, 0] < w["X_BBOX"][1])
            & (points[:, 1] > w["Y_BBOX"][0]) & (points[:, 1] < w["Y_BBOX"][1])
            & (points[:, 2] > w["Z_BBOX"][0]) & (points[:, 2] < w["Z_BBOX"][1])
        )
        points = points[keep]
    if len(points) == 0:
        raise ValueError("point cloud is empty after depth and workspace filtering")

    voxel = np.floor(points[:, :3] / calibration.voxel_size_m).astype(np.int64)
    _, keep_indices = np.unique(voxel, axis=0, return_index=True)
    points = points[np.sort(keep_indices)]
    limit = calibration.max_points_before_training
    if len(points) > limit:
        # Evenly spaced deterministic selection keeps offline and online preprocessing identical.
        points = points[np.linspace(0, len(points) - 1, limit, dtype=np.int64)]
    return np.ascontiguousarray(points, dtype=np.float32)
