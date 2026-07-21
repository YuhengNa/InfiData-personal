#!/usr/bin/env python3
"""Convert EgoVerse Zarr v3 episodes to an InfiData-style dataset.

This converter intentionally preserves EgoVerse source fields instead of
renaming them into another dataset's state/action schema. The output follows
the InfiData storage layout:

  out_root/
    data/egoverse/chunk-000/episode_000000.parquet
    meta/episodes.jsonl
    meta/segments.jsonl
    meta/tasks.json
    meta/robots.json
    meta/stats.json
    meta/source_metadata/<episode_id>.json

The script contains a small Zarr v3 sharding-indexed reader for the codecs
used by the downloaded EgoVerse data: numeric bytes+zstd arrays and
variable-length bytes+zstd arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import zstandard as zstd


U64_MAX = (1 << 64) - 1
DEFAULT_ACTION_SOURCE_HORIZON = 30
DEFAULT_ACTION_CHUNK_LENGTH = 100
STATE_KEY = "observations.state.ee_pose"
ACTION_CHUNK_KEY = "actions_cartesian"
CONTROL_MODE = "egoverse_cartesian"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"__bytes_len__": len(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def speed_bin_from_t(num_frames: int, bin_size: int = 500) -> int:
    return int(round(num_frames / bin_size) * bin_size)


def infer_action_stride(embodiment: str) -> int:
    normalized = embodiment.lower()
    if normalized.startswith("aria"):
        return 3
    if normalized.startswith(("mecka", "scale")):
        return 1
    return 3


def episode_chunk(episode_index: int, chunk_size: int = 1000) -> int:
    return episode_index // chunk_size


def dtype_from_zarr(data_type: str) -> np.dtype:
    if data_type == "float64":
        return np.dtype("<f8")
    if data_type == "float32":
        return np.dtype("<f4")
    if data_type == "int64":
        return np.dtype("<i8")
    if data_type == "int32":
        return np.dtype("<i4")
    if data_type == "uint8":
        return np.dtype("u1")
    raise ValueError(f"Unsupported numeric Zarr data_type: {data_type}")


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def ceil_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a + b - 1) // b


def unravel_index(index: int, shape: list[int]) -> tuple[int, ...]:
    coords = []
    for size in reversed(shape):
        coords.append(index % size)
        index //= size
    return tuple(reversed(coords))


def _xyzwxyz_to_matrix(xyzwxyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(xyzwxyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != 7:
        raise ValueError(f"Expected (N, 7) xyz+quat(wxyz), got {arr.shape}")

    mats = np.broadcast_to(np.eye(4, dtype=np.float64), (arr.shape[0], 4, 4)).copy()
    quat_xyzw = arr[:, [4, 5, 6, 3]]
    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Found zero-norm quaternion in EgoVerse pose.")
    quat_xyzw = quat_xyzw / norms
    mats[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
    mats[:, :3, 3] = arr[:, :3]
    return mats


def _xyzypr_to_matrix(xyzypr: np.ndarray) -> np.ndarray:
    arr = np.asarray(xyzypr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != 6:
        raise ValueError(f"Expected (N, 6) xyz+ypr, got {arr.shape}")

    mats = np.broadcast_to(np.eye(4, dtype=np.float64), (arr.shape[0], 4, 4)).copy()
    mats[:, :3, :3] = R.from_euler("ZYX", arr[:, 3:6], degrees=False).as_matrix()
    mats[:, :3, 3] = arr[:, :3]
    return mats


def _matrix_to_xyzwxyz(mats: np.ndarray) -> np.ndarray:
    arr = np.asarray(mats, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (N, 4, 4) matrices, got {arr.shape}")
    quat_xyzw = R.from_matrix(arr[:, :3, :3]).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([arr[:, :3, 3], quat_wxyz], axis=-1)


def _matrix_to_xyzypr(mats: np.ndarray) -> np.ndarray:
    arr = np.asarray(mats, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (N, 4, 4) matrices, got {arr.shape}")
    ypr = R.from_matrix(arr[:, :3, :3]).as_euler("ZYX", degrees=False)
    return np.concatenate([arr[:, :3, 3], ypr], axis=-1)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] == 7:
        return _xyzwxyz_to_matrix(arr)
    if arr.shape[-1] == 6:
        return _xyzypr_to_matrix(arr)
    raise ValueError(f"Expected pose dim 6 or 7, got {arr.shape}")


def transform_pose_chunk_to_target_frame(
    target_world_pose: np.ndarray | None,
    pose_world_chunk: np.ndarray,
) -> np.ndarray:
    poses = np.asarray(pose_world_chunk, dtype=np.float64)
    original_shape = poses.shape
    if poses.shape[-1] != 7:
        raise ValueError(f"Expected xyz+quat(wxyz) action poses, got {poses.shape}")
    flat_poses = poses.reshape(-1, 7)

    pose_mats = _xyzwxyz_to_matrix(flat_poses)
    if target_world_pose is not None:
        target_mat = pose_to_matrix(np.asarray(target_world_pose, dtype=np.float64))[0]
        pose_mats = np.linalg.inv(target_mat)[None, :, :] @ pose_mats
    return _matrix_to_xyzwxyz(pose_mats).reshape(original_shape)


def quat_pose_to_xyzypr(pose_or_chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose_or_chunk, dtype=np.float64)
    original_shape = arr.shape
    if arr.shape[-1] != 7:
        raise ValueError(f"Expected xyz+quat(wxyz), got {arr.shape}")
    mats = _xyzwxyz_to_matrix(arr.reshape(-1, 7))
    return _matrix_to_xyzypr(mats).reshape(*original_shape[:-1], 6)


def interpolate_quat_pose_chunk(
    chunk: np.ndarray,
    chunk_length: int,
    stride: int,
) -> np.ndarray:
    poses = np.asarray(chunk, dtype=np.float64)[::stride]
    if poses.ndim != 2 or poses.shape[-1] != 7:
        raise ValueError(f"Expected (T, 7) pose chunk, got {poses.shape}")
    if poses.shape[0] == 0:
        raise ValueError("Cannot interpolate an empty pose chunk.")
    if poses.shape[0] == 1:
        return np.repeat(poses[:1], chunk_length, axis=0)

    old_time = np.linspace(0, 1, poses.shape[0])
    new_time = np.linspace(0, 1, chunk_length)
    xyz = interp1d(old_time, poses[:, :3], axis=0, kind="linear")(new_time)

    quat_xyzw = poses[:, [4, 5, 6, 3]]
    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Found zero-norm quaternion in EgoVerse action chunk.")
    quat_xyzw = quat_xyzw / norms
    for i in range(1, quat_xyzw.shape[0]):
        if np.dot(quat_xyzw[i - 1], quat_xyzw[i]) < 0:
            quat_xyzw[i] = -quat_xyzw[i]

    if quat_xyzw.shape[0] == 1:
        quat_interp_xyzw = np.repeat(quat_xyzw[:1], chunk_length, axis=0)
    else:
        quat_interp_xyzw = Slerp(old_time, R.from_quat(quat_xyzw))(new_time).as_quat()
    quat_interp_wxyz = quat_interp_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([xyz, quat_interp_wxyz], axis=-1)


def build_cartesian_fields(
    arrays: dict[str, Any],
    frame_index: int,
    total_frames: int,
    action_source_horizon: int,
    action_chunk_length: int,
    action_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    required = ("left.obs_ee_pose", "right.obs_ee_pose")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Missing required cartesian source field(s): {missing}")

    left = np.asarray(arrays["left.obs_ee_pose"], dtype=np.float64)
    right = np.asarray(arrays["right.obs_ee_pose"], dtype=np.float64)
    if left.shape[0] < total_frames or right.shape[0] < total_frames:
        raise ValueError("left/right.obs_ee_pose is shorter than total_frames.")
    if left.shape[-1] != 7 or right.shape[-1] != 7:
        raise ValueError(
            f"Expected left/right.obs_ee_pose last dim 7, got {left.shape}/{right.shape}"
        )

    target_pose = None
    if "obs_head_pose" in arrays:
        head = np.asarray(arrays["obs_head_pose"], dtype=np.float64)
        if head.shape[0] >= total_frames:
            target_pose = head[frame_index]

    source_indices = np.minimum(
        np.arange(frame_index, frame_index + action_source_horizon),
        total_frames - 1,
    )
    left_action_head = transform_pose_chunk_to_target_frame(target_pose, left[source_indices])
    right_action_head = transform_pose_chunk_to_target_frame(target_pose, right[source_indices])
    left_obs_head = transform_pose_chunk_to_target_frame(target_pose, left[frame_index])
    right_obs_head = transform_pose_chunk_to_target_frame(target_pose, right[frame_index])

    left_action_interp = interpolate_quat_pose_chunk(
        left_action_head,
        chunk_length=action_chunk_length,
        stride=action_stride,
    )
    right_action_interp = interpolate_quat_pose_chunk(
        right_action_head,
        chunk_length=action_chunk_length,
        stride=action_stride,
    )

    left_action_ypr = quat_pose_to_xyzypr(left_action_interp)
    right_action_ypr = quat_pose_to_xyzypr(right_action_interp)
    left_state_ypr = quat_pose_to_xyzypr(left_obs_head)
    right_state_ypr = quat_pose_to_xyzypr(right_obs_head)

    state = np.concatenate([left_state_ypr, right_state_ypr], axis=-1)
    actions = np.concatenate([left_action_ypr, right_action_ypr], axis=-1)
    return state.astype(np.float32), actions.astype(np.float32)


class EgoVerseArray:
    def __init__(self, path: Path):
        self.path = path
        with (path / "zarr.json").open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.shape = [int(x) for x in self.meta.get("shape", [])]
        self.data_type = self.meta["data_type"]
        self.outer_chunk_shape = [
            int(x)
            for x in self.meta["chunk_grid"]["configuration"]["chunk_shape"]
        ]
        shard_codec = self.meta["codecs"][0]
        self.is_sharded = shard_codec["name"] == "sharding_indexed"
        if self.is_sharded:
            self.shard_config = shard_codec["configuration"]
            self.inner_chunk_shape = [
                int(x) for x in self.shard_config["chunk_shape"]
            ]
        elif self.meta["chunk_grid"]["name"] == "regular":
            self.shard_config = None
            self.inner_chunk_shape = self.outer_chunk_shape
        else:
            raise ValueError(
                f"Unsupported codec/grid for {path}: "
                f"{shard_codec['name']}/{self.meta['chunk_grid']['name']}"
            )
        self.decompressor = zstd.ZstdDecompressor()

    def read(self) -> Any:
        if self.data_type == "variable_length_bytes":
            return self._read_vlen_bytes()
        return self._read_numeric()

    def _chunk_files(self) -> list[Path]:
        cdir = self.path / "c"
        if not cdir.exists():
            return []
        return sorted(p for p in cdir.glob("**/*") if p.is_file())

    def _chunk_coords_from_path(self, chunk_file: Path) -> tuple[int, ...]:
        rel = chunk_file.relative_to(self.path / "c")
        return tuple(int(part) for part in rel.parts)

    def _inner_grid_shape(self) -> list[int]:
        return [
            ceil_div(outer, inner)
            for outer, inner in zip(self.outer_chunk_shape, self.inner_chunk_shape)
        ]

    def _read_indexed_entries(self, blob: bytes) -> list[tuple[int, int]]:
        n_entries = product(self._inner_grid_shape())
        if n_entries == 0:
            return []
        index_nbytes = n_entries * 16 + 4
        if len(blob) < index_nbytes:
            raise ValueError(f"Shard is too small for index: {self.path}")
        index_start = len(blob) - index_nbytes
        entries = []
        for i in range(n_entries):
            start = index_start + i * 16
            offset, nbytes = struct.unpack("<QQ", blob[start : start + 16])
            entries.append((offset, nbytes))
        return entries

    def _iter_decoded_inner_chunks(self):
        if not self.is_sharded:
            for chunk_file in self._chunk_files():
                chunk_coords = self._chunk_coords_from_path(chunk_file)
                raw = self.decompressor.decompress(chunk_file.read_bytes())
                yield chunk_coords, tuple(0 for _ in self.shape), raw
            return

        inner_grid = self._inner_grid_shape()
        for chunk_file in self._chunk_files():
            outer_coords = self._chunk_coords_from_path(chunk_file)
            blob = chunk_file.read_bytes()
            for i, (offset, nbytes) in enumerate(self._read_indexed_entries(blob)):
                if nbytes == 0 or offset == U64_MAX or nbytes == U64_MAX:
                    continue
                raw = self.decompressor.decompress(blob[offset : offset + nbytes])
                inner_coords = unravel_index(i, inner_grid)
                yield outer_coords, inner_coords, raw

    def _read_numeric(self) -> np.ndarray:
        dtype = dtype_from_zarr(self.data_type)
        if any(dim == 0 for dim in self.shape):
            return np.empty(self.shape, dtype=dtype)

        out = np.zeros(self.shape, dtype=dtype)
        for outer_coords, inner_coords, raw in self._iter_decoded_inner_chunks():
            start = [
                outer_coords[d] * self.outer_chunk_shape[d]
                + inner_coords[d] * self.inner_chunk_shape[d]
                for d in range(len(self.shape))
            ]
            stop = [
                min(start[d] + self.inner_chunk_shape[d], self.shape[d])
                for d in range(len(self.shape))
            ]
            if any(start[d] >= self.shape[d] for d in range(len(self.shape))):
                continue

            expected_count = product(self.inner_chunk_shape)
            arr = np.frombuffer(raw, dtype=dtype)
            valid_shape = tuple(stop[d] - start[d] for d in range(len(self.shape)))
            if arr.size == expected_count:
                arr = arr.reshape(self.inner_chunk_shape)
            elif arr.size == product(list(valid_shape)):
                arr = arr.reshape(valid_shape)
            else:
                raise ValueError(
                    f"Decoded chunk size mismatch for {self.path}: "
                    f"got {arr.size}, expected {expected_count} or {product(list(valid_shape))}"
                )

            source_slices = tuple(slice(0, stop[d] - start[d]) for d in range(len(self.shape)))
            target_slices = tuple(slice(start[d], stop[d]) for d in range(len(self.shape)))
            out[target_slices] = arr[source_slices]
        return out

    @staticmethod
    def _decode_vlen_payload(raw: bytes) -> list[bytes]:
        if len(raw) < 4:
            return []
        count = struct.unpack("<I", raw[:4])[0]
        pos = 4
        items = []
        for _ in range(count):
            if pos + 4 > len(raw):
                raise ValueError("Malformed vlen payload: missing item length")
            nbytes = struct.unpack("<I", raw[pos : pos + 4])[0]
            pos += 4
            items.append(raw[pos : pos + nbytes])
            pos += nbytes
        return items

    def _read_vlen_bytes(self) -> list[bytes]:
        if len(self.shape) != 1:
            raise ValueError(f"Only 1D vlen arrays are supported: {self.path}")
        total = self.shape[0]
        if total == 0:
            return []
        out: list[bytes | None] = [None] * total
        for outer_coords, inner_coords, raw in self._iter_decoded_inner_chunks():
            start = (
                outer_coords[0] * self.outer_chunk_shape[0]
                + inner_coords[0] * self.inner_chunk_shape[0]
            )
            items = self._decode_vlen_payload(raw)
            for j, item in enumerate(items):
                idx = start + j
                if idx < total:
                    out[idx] = item
        return [item if item is not None else b"" for item in out]


def load_episode(episode_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with (episode_path / "zarr.json").open("r", encoding="utf-8") as f:
        root_meta = json.load(f)

    arrays = {}
    features = root_meta["attributes"].get("features", {})
    for field_name in sorted(features):
        field_path = episode_path / field_name
        if not (field_path / "zarr.json").exists():
            continue
        arrays[field_name] = EgoVerseArray(field_path).read()
    return root_meta, arrays


def decode_annotations(values: list[bytes]) -> list[dict[str, Any]]:
    annotations = []
    for value in values:
        if not value:
            continue
        decoded = json.loads(value.decode("utf-8"))
        if isinstance(decoded, dict):
            annotations.append(decoded)
    return annotations


def annotations_for_frame(
    annotations: list[dict[str, Any]],
    frame_index: int,
) -> tuple[list[int], list[str], list[dict[str, Any]]]:
    indices = []
    texts = []
    records = []
    for i, ann in enumerate(annotations):
        start = int(ann.get("start_idx", -1))
        end = int(ann.get("end_idx", -1))
        if start <= frame_index < end:
            indices.append(i)
            text = str(ann.get("text", ""))
            texts.append(text)
            records.append(ann)
    return indices, texts, records


def value_at_frame(value: Any, frame_index: int) -> Any:
    if isinstance(value, np.ndarray):
        item = value[frame_index]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return item
    if isinstance(value, list):
        return value[frame_index]
    return value


def numeric_stats(arr: np.ndarray, total_frames: int) -> dict[str, Any]:
    sliced = arr[:total_frames]
    if sliced.ndim == 1:
        data = sliced.reshape(-1, 1)
    else:
        data = sliced.reshape(sliced.shape[0], -1)
    return {
        "shape": list(arr.shape),
        "valid_shape": list(sliced.shape),
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
    }


def convert_episode(
    episode_path: Path,
    out_root: Path,
    episode_index: int,
    overwrite: bool,
    action_source_horizon: int,
    action_chunk_length: int,
    action_stride: int | None,
    source_episode_index: int | None = None,
) -> dict[str, Any]:
    source_episode_id = episode_path.name
    source_index = int(episode_index if source_episode_index is None else source_episode_index)
    root_meta, arrays = load_episode(episode_path)
    attrs = root_meta["attributes"]
    features = attrs.get("features", {})
    total_frames = int(attrs["total_frames"])
    fps = int(attrs.get("fps", 30))
    task = str(attrs.get("task_description") or attrs.get("task_name") or "egoverse task")
    task_name = str(attrs.get("task_name") or task)
    embodiment = str(attrs.get("embodiment", "unknown"))
    episode_action_stride = (
        int(action_stride) if action_stride is not None else infer_action_stride(embodiment)
    )
    speed_bin = speed_bin_from_t(total_frames)

    annotations = decode_annotations(arrays.get("annotations", []))
    has_head_pose = "obs_head_pose" in arrays

    rows = []
    for t in range(total_frames):
        active_indices, active_texts, active_records = annotations_for_frame(annotations, t)
        subtask = active_texts[0] if active_texts else task
        state, actions_cartesian = build_cartesian_fields(
            arrays=arrays,
            frame_index=t,
            total_frames=total_frames,
            action_source_horizon=action_source_horizon,
            action_chunk_length=action_chunk_length,
            action_stride=episode_action_stride,
        )
        row: dict[str, Any] = {
            "episode_index": int(episode_index),
            "source_episode_index": int(source_index),
            "source_episode_id": source_episode_id,
            "frame_index": int(t),
            "timestamp": float(t / fps),
            "task": task,
            "task_name": task_name,
            "subtask": subtask,
            "prompt": task,
            "memory": subtask,
            "embodiment": embodiment,
            "robot_type": embodiment,
            "control_mode": CONTROL_MODE,
            "quality": 5,
            "speed_bin": int(speed_bin),
            "mistake": False,
            "success": True,
            "source_dataset": "EgoVerse",
            "domain": "real",
            "observation.state": state.tolist(),
            STATE_KEY: state.tolist(),
            "action": actions_cartesian[0].tolist(),
            ACTION_CHUNK_KEY: actions_cartesian.tolist(),
            "action_source_horizon": int(action_source_horizon),
            "action_chunk_length": int(action_chunk_length),
            "action_stride": int(episode_action_stride),
            "cartesian_frame": "obs_head_pose" if has_head_pose else "source_pose_frame",
            "cartesian_rotation": "yaw_pitch_roll",
            "cartesian_layout": "left_xyz_ypr_right_xyz_ypr",
            "annotations.active_indices": active_indices,
            "annotations.active_texts": active_texts,
            "annotations.active_records_json": json.dumps(active_records, ensure_ascii=False),
        }

        for field_name, value in arrays.items():
            if field_name == "annotations":
                continue
            source_dtype = features.get(field_name, {}).get("dtype")
            if isinstance(value, np.ndarray):
                if value.shape and value.shape[0] >= total_frames:
                    row[field_name] = value_at_frame(value, t)
            elif isinstance(value, list):
                if len(value) >= total_frames:
                    if source_dtype == "jpeg":
                        row[field_name] = bytes(value[t])
                    else:
                        row[field_name] = value_at_frame(value, t)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    chunk_index = episode_chunk(episode_index)
    parquet_rel_path = (
        Path("data")
        / "egoverse"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    parquet_path = out_root / parquet_rel_path
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists() and not overwrite:
        raise FileExistsError(f"Output parquet exists: {parquet_path}")
    out_df.to_parquet(parquet_path, index=False, compression="zstd")

    source_meta_rel_path = (
        Path("meta") / "source_metadata" / f"{source_episode_id}.json"
    )
    write_json(
        out_root / source_meta_rel_path,
        {
            "episode_id": source_episode_id,
            "root_zarr_json": root_meta,
            "field_zarr_json": {
                field: json.loads((episode_path / field / "zarr.json").read_text(encoding="utf-8"))
                for field in arrays
                if (episode_path / field / "zarr.json").exists()
            },
        },
    )

    preserved_fields = []
    field_stats = {}
    byte_fields = {}
    for field_name, value in arrays.items():
        spec = dict(features.get(field_name, {}))
        spec["storage_column"] = None if field_name == "annotations" else field_name
        preserved_fields.append({"field": field_name, **spec})
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] >= total_frames:
            field_stats[field_name] = numeric_stats(value, total_frames)
        elif isinstance(value, list):
            byte_fields[field_name] = {
                "num_items": len(value),
                "total_bytes": int(sum(len(item) for item in value)),
                "max_item_bytes": int(max((len(item) for item in value), default=0)),
            }

    episode_meta = {
        "episode_index": int(episode_index),
        "source_episode_index": int(source_index),
        "source_episode_id": source_episode_id,
        "parquet_path": str(parquet_rel_path),
        "source_metadata_path": str(source_meta_rel_path),
        "task": task,
        "task_name": task_name,
        "embodiment": embodiment,
        "robot_type": embodiment,
        "control_mode": CONTROL_MODE,
        "num_frames": int(total_frames),
        "fps": int(fps),
        "duration_sec": float(total_frames / fps),
        "speed_bin": int(speed_bin),
        "quality": 5,
        "success": True,
        "mistake": False,
        "source_dataset": "EgoVerse",
        "domain": "real",
        "preserved_fields": preserved_fields,
        "image_fields": [
            name for name, spec in features.items() if spec.get("dtype") == "jpeg"
        ],
        "derived_fields": [
            {
                "field": "observation.state",
                "shape": [12],
                "dtype": "float32",
                "description": "Alias of observations.state.ee_pose for LeRobot/openpi conversion.",
            },
            {
                "field": STATE_KEY,
                "shape": [12],
                "dtype": "float32",
                "description": "Current left/right end-effector poses in cartesian mode: [left xyz ypr, right xyz ypr].",
            },
            {
                "field": "action",
                "shape": [12],
                "dtype": "float32",
                "description": "First step of actions_cartesian; kept for simple LeRobot-style per-frame action schemas.",
            },
            {
                "field": ACTION_CHUNK_KEY,
                "shape": [action_chunk_length, 12],
                "dtype": "float32",
                "description": "Future observed left/right ee poses transformed into the current head frame when available, then interpolated to a fixed-length chunk.",
            },
        ],
        "state_key": STATE_KEY,
        "action_key": ACTION_CHUNK_KEY,
        "lerobot_state_key": "observation.state",
        "lerobot_action_key": "action",
        "state_dim": 12,
        "action_dim": 12,
        "action_chunk_length": int(action_chunk_length),
        "action_source_horizon": int(action_source_horizon),
        "action_stride": int(episode_action_stride),
        "cartesian_frame": "obs_head_pose" if has_head_pose else "source_pose_frame",
        "cartesian_rotation": "yaw_pitch_roll",
        "cartesian_layout": "left_xyz_ypr_right_xyz_ypr",
        "annotation_count": len(annotations),
        "root_attributes": attrs,
    }

    if annotations:
        segments = [
            {
                "episode_index": int(episode_index),
                "segment_index": int(i),
                "start_frame": int(ann["start_idx"]),
                "end_frame": int(min(max(int(ann["end_idx"]) - 1, int(ann["start_idx"])), total_frames - 1)),
                "task": task,
                "subtask": str(ann.get("text", "")),
                "mistake": False,
                "annotation_status": "source_annotation_v1",
                "source_annotation_status": "egoverse_annotation_v1",
                "source_segment_index": int(i),
                "source_frame_count": int(
                    min(max(int(ann["end_idx"]) - int(ann["start_idx"]), 1), total_frames)
                ),
                "source_episode_index": int(source_index),
                "source_episode_id": source_episode_id,
                "source_annotation": to_jsonable(ann),
            }
            for i, ann in enumerate(annotations)
        ]
    else:
        segments = [
            {
                "episode_index": int(episode_index),
                "segment_index": 0,
                "start_frame": 0,
                "end_frame": int(total_frames - 1),
                "task": task,
                "subtask": task,
                "mistake": False,
                "annotation_status": "empty_source_annotations",
                "source_annotation_status": "empty_source_annotations",
                "source_segment_index": 0,
                "source_frame_count": int(total_frames),
                "source_episode_index": int(source_index),
                "source_episode_id": source_episode_id,
                "source_annotation": {},
            }
        ]

    stats = {
        "episode_index": int(episode_index),
        "source_episode_id": source_episode_id,
        "num_frames": int(total_frames),
        "numeric_fields": field_stats,
        "byte_fields": byte_fields,
    }
    return {
        "episode_meta": episode_meta,
        "segments": segments,
        "stats": stats,
    }


def write_dataset_meta(out_root: Path, converted: list[dict[str, Any]]) -> None:
    episode_meta = [item["episode_meta"] for item in converted]
    segments = [segment for item in converted for segment in item["segments"]]
    episode_stats = [item["stats"] for item in converted]

    tasks = {
        str(i): {"task": task, "source_dataset": "EgoVerse"}
        for i, task in enumerate(sorted({item["task"] for item in episode_meta}))
    }
    robots = {}
    for item in episode_meta:
        robot = item["robot_type"]
        robots[robot] = {
            "robot_type": robot,
            "domain": "real",
            "control_mode": CONTROL_MODE,
            "fps": int(item["fps"]),
            "image_fields": item["image_fields"],
            "state_key": item["state_key"],
            "action_key": item["action_key"],
            "state_dim": item["state_dim"],
            "action_dim": item["action_dim"],
            "action_chunk_length": item["action_chunk_length"],
            "cartesian_frame": item["cartesian_frame"],
            "cartesian_rotation": item["cartesian_rotation"],
        }

    dataset_stats = {
        "num_episodes": len(episode_meta),
        "num_frames": int(sum(item["num_frames"] for item in episode_meta)),
        "episodes": episode_stats,
    }
    action_chunk_length = int(episode_meta[0]["action_chunk_length"])

    write_jsonl(out_root / "meta" / "episodes.jsonl", episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", segments)
    write_json(out_root / "meta" / "tasks.json", tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", dataset_stats)

    readme = f"""# EgoVerse InfiData

Converted from EgoVerse Zarr v3 episodes.

- Episodes: {len(episode_meta)}
- Frames: {dataset_stats["num_frames"]}
- Storage: one parquet per episode under `data/egoverse/`
- Field policy: EgoVerse source fields are preserved as parquet columns, with cartesian training fields added.
- Images: source JPEG bytes are stored directly in their original `images.*` columns.
- Annotations: source `annotation_v1` records are stored in `meta/segments.jsonl`; frame-level active annotation columns are added to parquet.
- Cartesian state: `observation.state` and `{STATE_KEY}` are `[left xyz ypr, right xyz ypr]`.
- Cartesian action: `action` is the first 12D point; `{ACTION_CHUNK_KEY}` is a [{action_chunk_length}, 12] future observed EE-pose chunk.
- Source Zarr metadata: preserved under `meta/source_metadata/`.
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")


def discover_episode_paths(input_root: Path, episode_id: str | None) -> list[Path]:
    if episode_id:
        return [input_root / episode_id]
    return sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and (path / "zarr.json").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EgoVerse Zarr episodes into an InfiData-style dataset."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/mnt/workspace/wudi/ELUBrain/EgoVerseData"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/workspace/InfiData/EgoVerse"),
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Episode folder name to convert. Omit to convert discovered episodes in sorted order.",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Output episode index for --episode-id mode.")
    parser.add_argument("--start-episode-index", type=int, default=0, help="Output episode index offset for batch mode.")
    parser.add_argument(
        "--input-start-index",
        type=int,
        default=0,
        help="Start offset in the sorted input episode list for batch mode.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of discovered episodes to convert.")
    parser.add_argument("--action-source-horizon", type=int, default=DEFAULT_ACTION_SOURCE_HORIZON)
    parser.add_argument("--action-chunk-length", type=int, default=DEFAULT_ACTION_CHUNK_LENGTH)
    parser.add_argument(
        "--action-stride",
        type=int,
        default=None,
        help="Override per-episode stride. Default: infer from embodiment, e.g. aria=3, mecka/scale=1.",
    )
    parser.add_argument("--clean-output", action="store_true", help="Delete out-root before converting.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail immediately instead of skipping invalid episodes.")
    args = parser.parse_args()

    if args.action_source_horizon <= 0:
        raise ValueError("--action-source-horizon must be positive")
    if args.action_chunk_length <= 0:
        raise ValueError("--action-chunk-length must be positive")
    if args.action_stride is not None and args.action_stride <= 0:
        raise ValueError("--action-stride must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if args.input_start_index < 0:
        raise ValueError("--input-start-index must be non-negative")
    if args.clean_output and args.out_root.exists():
        shutil.rmtree(args.out_root)

    args.out_root.mkdir(parents=True, exist_ok=True)
    episode_paths = discover_episode_paths(args.input_root, args.episode_id)
    input_start_index = 0 if args.episode_id else args.input_start_index
    if input_start_index:
        episode_paths = episode_paths[input_start_index:]
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"No episode folders found under {args.input_root}")

    converted = []
    errors = []
    start = time.perf_counter()
    for local_i, episode_path in enumerate(episode_paths):
        source_episode_index = args.episode_index if args.episode_id else input_start_index + local_i
        episode_index = (
            args.episode_index
            if args.episode_id
            else args.start_episode_index + len(converted)
        )
        try:
            if not episode_path.is_dir():
                raise FileNotFoundError(f"Episode folder not found: {episode_path}")
            item = convert_episode(
                episode_path=episode_path,
                out_root=args.out_root,
                episode_index=episode_index,
                overwrite=args.overwrite,
                action_source_horizon=args.action_source_horizon,
                action_chunk_length=args.action_chunk_length,
                action_stride=args.action_stride,
                source_episode_index=source_episode_index,
            )
            converted.append(item)
            print(
                f"[OK] {len(converted)}/{len(episode_paths)} "
                f"episode_index={episode_index} source_index={source_episode_index} "
                f"source={episode_path.name}"
            )
        except Exception as exc:
            error = {
                "episode_path": str(episode_path),
                "episode_index": int(episode_index),
                "source_episode_index": int(source_episode_index),
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(f"[SKIP] episode_index={episode_index} source={episode_path.name}: {error['error']}")
            if args.strict:
                raise

    if not converted:
        write_jsonl(args.out_root / "meta" / "conversion_errors.jsonl", errors)
        raise RuntimeError("No episodes were converted successfully.")
    write_dataset_meta(args.out_root, converted)
    if errors:
        write_jsonl(args.out_root / "meta" / "conversion_errors.jsonl", errors)
    elapsed = time.perf_counter() - start
    print(
        f"[DONE] Converted {len(converted)} episode(s), skipped {len(errors)}, "
        f"elapsed {elapsed:.1f}s, output={args.out_root}"
    )


if __name__ == "__main__":
    main()
