#!/usr/bin/env python3
"""Convert a recorder LeRobot-v3 RGB-D dataset to PointAct's v2.1 + LMDB layout.

Run this with the PointAct environment (LeRobot 0.3.x). The source is parsed directly, so
the script does not import the newer LeRobot version which created it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
from itertools import islice
import json
import shutil
from pathlib import Path

import av
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pointact.utils.franka_rgbd import FrankaRGBDCalibration, rgbd_to_point_cloud, unpack_recorded_depth

msgpack_numpy.patch()

RGB_KEY = "observation.images.d455"
DEPTH_KEY = "observation.images.d455_depth"


def iter_parquet_rows(source: Path):
    files = sorted((source / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files below {source / 'data'}")
    expected_index = 0
    columns = ["observation.state", "action", DEPTH_KEY, "task_index", "episode_index", "frame_index", "index"]
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=16):
            for row in batch.to_pylist():
                if int(row["index"]) != expected_index:
                    raise ValueError(f"non-contiguous source index in {path}: {row['index']} != {expected_index}")
                expected_index += 1
                yield row


def iter_rgb_frames(source: Path):
    paths = sorted((source / "videos" / RGB_KEY).glob("**/*.mp4"))
    if not paths:
        raise FileNotFoundError(f"no D455 RGB videos below {source / 'videos' / RGB_KEY}")
    for path in paths:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")


def load_tasks(source: Path) -> dict[int, str]:
    table = pq.read_table(source / "meta" / "tasks.parquet")
    rows = table.to_pylist()
    tasks = {}
    for index, row in enumerate(rows):
        task_index = int(row.get("task_index", index))
        task = row.get("task") or row.get("tasks")
        if isinstance(task, list):
            task = task[0]
        tasks[task_index] = str(task)
    return tasks


def image_from_struct(value) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"expected embedded depth image struct, got {type(value).__name__}")
    if value.get("bytes") is not None:
        return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    if value.get("path"):
        return np.asarray(Image.open(value["path"]).convert("RGB"))
    raise ValueError("depth image has neither embedded bytes nor path")


def preprocess_frame(item, calibration: FrankaRGBDCalibration):
    """Decode one depth PNG and build its point cloud; safe to run in a worker thread."""
    row, rgb = item
    depth_mm = unpack_recorded_depth(image_from_struct(row[DEPTH_KEY]))
    return row, rgb, rgbd_to_point_cloud(rgb, depth_mm, calibration)


def iter_preprocessed_frames(rows, rgbs, calibration, workers: int, batch_size: int):
    """Preprocess bounded batches concurrently while preserving source frame order."""
    rgb_count = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="franka-rgbd") as executor:
        while True:
            row_batch = list(islice(rows, batch_size))
            if not row_batch:
                return
            items = []
            for row in row_batch:
                try:
                    rgb = next(rgbs)
                except StopIteration as exc:
                    raise ValueError(f"D455 RGB video ended after {rgb_count} frames") from exc
                items.append((row, rgb))
                rgb_count += 1
            futures = [executor.submit(preprocess_frame, item, calibration) for item in items]
            for future in futures:
                yield future.result()


def output_features(source_info: dict) -> dict:
    features = source_info["features"]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(features["observation.state"]["shape"]),
            "names": features["observation.state"].get("names"),
        },
        "action": {
            "dtype": "float32",
            "shape": tuple(features["action"]["shape"]),
            "names": features["action"].get("names"),
        },
        RGB_KEY: {
            "dtype": "video",
            "shape": tuple(features[RGB_KEY]["shape"]),
            "names": features[RGB_KEY].get("names"),
        },
    }


def convert(args: argparse.Namespace) -> None:
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    calibration = FrankaRGBDCalibration.load(args.calibration)
    with (source / "meta" / "info.json").open(encoding="utf-8") as f:
        info = json.load(f)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected a LeRobot v3.0 source, got {info.get('codebase_version')!r}")
    rgb_shape = tuple(info["features"][RGB_KEY]["shape"])
    depth_shape = tuple(info["features"][DEPTH_KEY]["shape"])
    expected_shape = (calibration.height, calibration.width, 3)
    if rgb_shape != expected_shape or depth_shape != expected_shape:
        raise ValueError(f"calibration expects {expected_shape}, dataset has RGB={rgb_shape}, depth={depth_shape}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)

    writer = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        fps=int(info["fps"]),
        robot_type=info.get("robot_type", "franka_fr3"),
        features=output_features(info),
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=args.image_writer_threads,
    )
    points_dir = output / args.point_cloud_dirname
    point_env = lmdb.open(str(points_dir), map_size=int(args.lmdb_map_size_gb * 1024**3), subdir=True)
    tasks = load_tasks(source)
    rows = iter_parquet_rows(source)
    rgbs = iter_rgb_frames(source)
    expected_total = int(info["total_frames"])
    current_episode = None
    converted = 0
    txn = point_env.begin(write=True)
    try:
        processed_frames = iter_preprocessed_frames(
            rows, rgbs, calibration, args.workers, args.preprocessing_batch_size
        )
        for row, rgb, points in tqdm(
            processed_frames, total=expected_total, unit="frame", desc="Converting Franka RGB-D"
        ):
            episode = int(row["episode_index"])
            if current_episode is not None and episode != current_episode:
                writer.save_episode()
            current_episode = episode
            point_key = f"{episode}-{int(row['frame_index'])}".encode("ascii")
            txn.put(point_key, msgpack.packb(points, use_bin_type=True))
            if converted and converted % args.commit_every == 0:
                txn.commit()
                txn = point_env.begin(write=True)
            task_index = int(row["task_index"])
            writer.add_frame(
                {
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    RGB_KEY: rgb,
                },
                task=tasks[task_index],
            )
            converted += 1
        if current_episode is not None:
            writer.save_episode()
        txn.commit()
        txn = None
        try:
            next(rgbs)
        except StopIteration:
            pass
        else:
            raise ValueError("D455 RGB video contains more frames than the parquet data")
        if converted != expected_total:
            raise ValueError(f"converted {converted} frames, expected {expected_total}")
        manifest = {
            "source": str(source),
            "frames": converted,
            "episodes": int(info["total_episodes"]),
            "fps": int(info["fps"]),
            "rgb_key": RGB_KEY,
            "depth_key": DEPTH_KEY,
            "point_cloud_dirname": args.point_cloud_dirname,
            "calibration": str(args.calibration.expanduser().resolve()),
            "calibration_sha256": calibration.fingerprint(),
        }
        with (output / "franka_rgbd_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Converted {converted} frames to {output}")
        print(f"Calibration fingerprint: {manifest['calibration_sha256']}")
    finally:
        if txn is not None:
            txn.abort()
        point_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="local PointAct/LeRobot dataset identifier")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--point-cloud-dirname", default="points_d455")
    parser.add_argument("--lmdb-map-size-gb", type=float, default=64.0)
    parser.add_argument("--commit-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4, help="bounded RGB-D preprocessing threads")
    parser.add_argument("--preprocessing-batch-size", type=int, default=16)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(
        args.lmdb_map_size_gb,
        args.commit_every,
        args.workers,
        args.preprocessing_batch_size,
        args.image_writer_threads,
    ) <= 0:
        parser.error("LMDB size, intervals, and worker counts must be positive")
    return args


if __name__ == "__main__":
    convert(parse_args())
