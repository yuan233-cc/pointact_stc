import base64

import numpy as np

from pointact.utils.franka_rgbd import (
    FrankaRGBDCalibration,
    decode_ros_depth,
    rgbd_to_point_cloud,
    unpack_recorded_depth,
    validated_target_pose,
)


def calibration() -> FrankaRGBDCalibration:
    return FrankaRGBDCalibration(
        width=2,
        height=2,
        intrinsics=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        camera_to_base=np.eye(4),
        min_depth_m=0.0,
        max_depth_m=2.0,
        voxel_size_m=0.0001,
        max_points_before_training=10,
    )


def test_validated_target_pose_normalizes_quaternion_and_checks_workspace():
    workspace = {"X_BBOX": [0.0, 1.0], "Y_BBOX": [-1.0, 1.0], "Z_BBOX": [0.0, 1.0]}
    pose = validated_target_pose(np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 2.0]), workspace)
    assert pose == [0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]


def test_unpack_recorded_depth_honors_validity_channel():
    packed = np.array([[[0x01, 0x02, 255], [0x03, 0x04, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(unpack_recorded_depth(packed), [[258, 0]])


def test_decode_ros_depth_preserves_padded_little_endian_rows():
    raw = b"\x01\x00\x02\x00xx\x03\x00\x04\x00yy"
    payload = {
        "height": 2,
        "width": 2,
        "step": 6,
        "encoding": "16UC1",
        "is_bigendian": 0,
        "data_b64": base64.b64encode(raw).decode(),
    }
    np.testing.assert_array_equal(decode_ros_depth(payload), [[1, 2], [3, 4]])


def test_rgbd_to_point_cloud_uses_camera_geometry_and_rgb_range():
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    depth_mm = np.full((2, 2), 1000, dtype=np.uint16)

    points = rgbd_to_point_cloud(rgb, depth_mm, calibration())

    np.testing.assert_allclose(points[:, :3], [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    np.testing.assert_allclose(points[:, 3:], rgb.reshape(-1, 3) / 255.0)
