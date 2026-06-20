"""Convert InfiData EgoVerse episodes to semantic-preserving RLDS/TFDS.

EgoVerse is ego/Cartesian data rather than joint robot control data. The source
stores per-frame JPEG bytes in parquet columns and stores Cartesian action
chunks as ``actions_cartesian`` with shape ``[action_chunk_length, action_dim]``.

This converter builds one RLDS dataset per stable ``robot_type + camera_set``
variant. It preserves source semantics instead of reshaping EgoVerse into a
DROID/RoboMIND/OpenPI-style robot schema.

Example smoke test:
    python scripts/convert2openpi/convert_infidata_egoverse_to_rlds.py \
        --infidata-root /mnt/workspace/InfiData/EgoVerse \
        --schema-key aria_bimanual \
        --data-dir /mnt/workspace/tmp/egoverse_aria_one \
        --max-episodes 1 \
        --overwrite

List variants:
    python scripts/convert2openpi/convert_infidata_egoverse_to_rlds.py \
        --infidata-root /mnt/workspace/InfiData/EgoVerse \
        --list-schemas
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
import tqdm

from video_decode_utils import append_skip


TEXT_COLUMNS = (
    "task",
    "task_name",
    "subtask",
    "prompt",
    "memory",
    "embodiment",
    "robot_type",
    "control_mode",
    "source_dataset",
    "domain",
    "cartesian_frame",
    "cartesian_rotation",
    "cartesian_layout",
)

STANDARD_COLUMNS = {
    "episode_index",
    "source_episode_id",
    "frame_index",
    "timestamp",
    "task",
    "task_name",
    "subtask",
    "prompt",
    "memory",
    "embodiment",
    "robot_type",
    "control_mode",
    "quality",
    "speed_bin",
    "mistake",
    "success",
    "source_dataset",
    "domain",
    "observation.state",
    "observations.state.ee_pose",
    "action",
    "actions_cartesian",
    "action_source_horizon",
    "action_chunk_length",
    "action_stride",
    "cartesian_frame",
    "cartesian_rotation",
    "cartesian_layout",
    "annotations.active_indices",
    "annotations.active_texts",
    "annotations.active_records_json",
}

MAX_TFDS_EXAMPLE_PARQUET_BYTES = int(1.8 * 1024**3)


@dataclasses.dataclass(frozen=True)
class InfiEpisode:
    episode_index: int
    source_episode_index: int
    source_episode_id: str
    parquet_path: str
    source_metadata_path: str
    raw_json: dict[str, Any]
    num_frames: int
    schema_key: str
    cameras: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SchemaVariant:
    schema_key: str
    cameras: tuple[str, ...]
    episode_count: int
    frame_count: int
    state_dim: int
    action_dim: int
    action_chunk_length: int
    fps: int

    @property
    def slug(self) -> str:
        return _slugify(f"{self.schema_key}_{'_'.join(self.cameras)}")


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "egoverse_schema"


def _feature_key(source_column: str) -> str:
    return _slugify(source_column)


def _as_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    return value if value else fallback


def _as_bool(value: Any) -> bool:
    if isinstance(value, np.bool_):
        return bool(value)
    return bool(value)


def _as_float_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(list(value), dtype=np.float32).reshape(shape)


def _as_float_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    return np.stack([np.asarray(item, dtype=np.float32) for item in value], axis=0).reshape(shape)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_robots_json(infidata_root: Path) -> dict[str, Any]:
    robots = _load_json(infidata_root / "meta" / "robots.json")
    if not isinstance(robots, dict):
        raise ValueError(f"Expected object in {infidata_root / 'meta' / 'robots.json'}")
    return robots


def _image_fields_from_episode(item: dict[str, Any]) -> tuple[str, ...]:
    fields = []
    for field in item.get("preserved_fields", []):
        name = str(field.get("field", ""))
        if name.startswith("images."):
            fields.append(name)
    if not fields:
        fields = [str(name) for name in item.get("image_fields", [])]
    return tuple(sorted(fields))


def _camera_name(image_field: str) -> str:
    return image_field.split(".", 1)[1] if image_field.startswith("images.") else image_field


def _load_all_episodes(infidata_root: Path, robots_json: dict[str, Any]) -> list[InfiEpisode]:
    episodes: list[InfiEpisode] = []
    with (infidata_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            schema_key = _as_text(item.get("robot_type") or item.get("embodiment"), "")
            if schema_key not in robots_json:
                raise KeyError(f"No robots.json schema for EgoVerse robot_type={schema_key!r}")
            episodes.append(
                InfiEpisode(
                    episode_index=int(item["episode_index"]),
                    source_episode_index=int(item.get("source_episode_index", -1)),
                    source_episode_id=_as_text(item.get("source_episode_id"), ""),
                    parquet_path=str(item["parquet_path"]),
                    source_metadata_path=_as_text(item.get("source_metadata_path"), ""),
                    raw_json=item,
                    num_frames=int(item.get("num_frames", 0)),
                    schema_key=schema_key,
                    cameras=tuple(_camera_name(field) for field in _image_fields_from_episode(item)),
                )
            )
    return episodes


def _schema_variants(episodes: list[InfiEpisode], robots_json: dict[str, Any]) -> list[SchemaVariant]:
    counts: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(lambda: [0, 0])
    for episode in episodes:
        key = (episode.schema_key, episode.cameras)
        counts[key][0] += 1
        counts[key][1] += episode.num_frames

    variants: list[SchemaVariant] = []
    for (schema_key, cameras), (episode_count, frame_count) in counts.items():
        config = robots_json[schema_key]
        variants.append(
            SchemaVariant(
                schema_key=schema_key,
                cameras=cameras,
                episode_count=episode_count,
                frame_count=frame_count,
                state_dim=int(config.get("state_dim", 0)),
                action_dim=int(config.get("action_dim", 0)),
                action_chunk_length=int(config.get("action_chunk_length", 0)),
                fps=int(config.get("fps", 0)),
            )
        )
    variants.sort(key=lambda item: (item.schema_key, item.cameras))
    return variants


def _print_schema_variants(variants: list[SchemaVariant]) -> None:
    print("Available EgoVerse RLDS variants:")
    for variant in variants:
        cameras = ",".join(variant.cameras) if variant.cameras else "<no_images>"
        print(
            f"- {variant.schema_key} | cameras={cameras} | episodes={variant.episode_count:,} | "
            f"frames={variant.frame_count:,} | state_dim={variant.state_dim} | action_dim={variant.action_dim} | "
            f"action_chunk_length={variant.action_chunk_length} | fps={variant.fps} | slug={variant.slug}"
        )


def _select_episodes(
    episodes: list[InfiEpisode],
    *,
    schema_key: str,
    cameras: tuple[str, ...],
    start_episode: int,
    max_episodes: int | None,
) -> list[InfiEpisode]:
    selected = [
        episode
        for episode in episodes
        if episode.schema_key == schema_key and episode.cameras == cameras and episode.episode_index >= start_episode
    ]
    selected.sort(key=lambda episode: episode.episode_index)
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return selected


def _sample_count(total: int, ratio: float) -> int:
    if not 0 <= ratio < 1:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    if total <= 1 or ratio == 0:
        return 0
    return min(max(int(round(total * ratio)), 1), total - 1)


def _split_episodes(
    episodes: list[InfiEpisode],
    *,
    unseen_test_ratio: float,
    seen_test_ratio: float,
    seed: int,
) -> tuple[list[InfiEpisode], list[InfiEpisode], list[InfiEpisode]]:
    if not episodes:
        return [], [], []
    if len(episodes) == 1:
        return episodes, [], []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    unseen_count = _sample_count(len(episodes), unseen_test_ratio)
    unseen_indices = set(int(index) for index in indices[:unseen_count])
    train_indices = [int(index) for index in indices if int(index) not in unseen_indices]

    seen_count = _sample_count(len(train_indices), seen_test_ratio)
    seen_indices = set(train_indices[:seen_count])

    train = [episode for index, episode in enumerate(episodes) if index not in unseen_indices]
    unseen_test = [episode for index, episode in enumerate(episodes) if index in unseen_indices]
    seen_test = [episode for index, episode in enumerate(episodes) if index in seen_indices]
    train.sort(key=lambda episode: episode.episode_index)
    unseen_test.sort(key=lambda episode: episode.episode_index)
    seen_test.sort(key=lambda episode: episode.episode_index)
    return train, unseen_test, seen_test


def _read_sample_rows(infidata_root: Path, episode: InfiEpisode) -> pd.DataFrame:
    return pd.read_parquet(infidata_root / episode.parquet_path).reset_index(drop=True)


def _parquet_columns(infidata_root: Path, episode: InfiEpisode) -> set[str]:
    return set(pq.ParquetFile(infidata_root / episode.parquet_path).schema_arrow.names)


def _parquet_size_bytes(infidata_root: Path, episode: InfiEpisode) -> int:
    return (infidata_root / episode.parquet_path).stat().st_size


def _image_shapes(episode: InfiEpisode, robot_config: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    shapes: dict[str, tuple[int, int, int]] = {}
    for field in episode.raw_json.get("preserved_fields", []):
        name = str(field.get("field", ""))
        if not name.startswith("images."):
            continue
        shape = tuple(int(dim) for dim in field.get("shape", []))
        if len(shape) != 3:
            raise ValueError(f"Expected HWC image shape for {name}, got {shape}")
        shapes[_camera_name(name)] = shape
    for name in robot_config.get("image_fields", []):
        camera = _camera_name(str(name))
        shapes.setdefault(camera, (480, 640, 3))
    return shapes


def _vector_columns(rows: pd.DataFrame) -> tuple[dict[str, tuple[int, ...]], dict[str, np.dtype]]:
    float_vectors: dict[str, tuple[int, ...]] = {}
    int_scalars: dict[str, np.dtype] = {}
    first = rows.iloc[0]
    for column in rows.columns:
        if column in STANDARD_COLUMNS or column.startswith("images."):
            continue
        value = first[column]
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(list(value))
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                float_vectors[column] = tuple(int(dim) for dim in arr.shape)
        elif isinstance(value, (int, np.integer)):
            int_scalars[column] = np.int64
    return float_vectors, int_scalars


def _annotation_indices_json(value: Any) -> str:
    if value is None:
        return "[]"
    try:
        return _json_dumps([int(item) for item in value])
    except TypeError:
        return "[]"


def _annotation_texts_json(value: Any) -> str:
    if value is None:
        return "[]"
    try:
        return _json_dumps([_as_text(item) for item in value])
    except TypeError:
        return "[]"


def _source_metadata_json(infidata_root: Path, episode: InfiEpisode) -> str:
    if not episode.source_metadata_path:
        return "{}"
    path = infidata_root / episode.source_metadata_path
    if not path.exists():
        return "{}"
    return path.read_text(encoding="utf-8")


def _state_action_schema(robot_config: dict[str, Any], image_fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        "robot_type": robot_config.get("robot_type", ""),
        "domain": robot_config.get("domain", "real"),
        "control_mode": robot_config.get("control_mode", "egoverse_cartesian"),
        "fps": int(robot_config.get("fps", 0)),
        "state_key": robot_config.get("state_key", "observations.state.ee_pose"),
        "state_representation": "bimanual_cartesian_ee_pose",
        "state_layout": "left_xyz_yaw_pitch_roll_right_xyz_yaw_pitch_roll",
        "state_dim": int(robot_config.get("state_dim", 12)),
        "action_key": robot_config.get("action_key", "actions_cartesian"),
        "action_representation": "absolute_cartesian_pose_chunk",
        "action_dim": int(robot_config.get("action_dim", 12)),
        "action_chunk_length": int(robot_config.get("action_chunk_length", 100)),
        "action_is_delta": False,
        "action_single_step_note": "The source action column equals actions_cartesian[0].",
        "action_chunk_note": (
            "actions_cartesian is preserved as a [chunk_length, action_dim] Cartesian pose chunk. "
            "The first element is the current Cartesian pose in the stated cartesian_frame."
        ),
        "cartesian_frame": robot_config.get("cartesian_frame", ""),
        "cartesian_rotation": robot_config.get("cartesian_rotation", "yaw_pitch_roll"),
        "image_fields": list(image_fields),
    }


class EgoVerseInfidata(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Semantic-preserving InfiData EgoVerse RLDS split by ego embodiment schema."}

    def __init__(
        self,
        *,
        infidata_root: Path,
        train_episodes: list[InfiEpisode],
        unseen_test_episodes: list[InfiEpisode],
        seen_test_episodes: list[InfiEpisode],
        schema_key: str,
        cameras: tuple[str, ...],
        image_shapes: dict[str, tuple[int, int, int]],
        state_dim: int,
        action_dim: int,
        action_chunk_length: int,
        robot_config: dict[str, Any],
        robots_json: dict[str, Any],
        float_vector_columns: dict[str, tuple[int, ...]],
        int_scalar_columns: dict[str, np.dtype],
        state_action_schema: dict[str, Any],
        skip_log_path: Path,
        **kwargs: Any,
    ):
        self._infidata_root = Path(infidata_root)
        self._train_episodes = train_episodes
        self._unseen_test_episodes = unseen_test_episodes
        self._seen_test_episodes = seen_test_episodes
        self._schema_key = schema_key
        self._cameras = cameras
        self._image_shapes = image_shapes
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._action_chunk_length = int(action_chunk_length)
        self._robot_config = robot_config
        self._robots_json = robots_json
        self._float_vector_columns = float_vector_columns
        self._int_scalar_columns = int_scalar_columns
        self._state_action_schema = state_action_schema
        self._skip_log_path = Path(skip_log_path)
        self._float_key_to_source = {_feature_key(column): column for column in float_vector_columns}
        self._int_key_to_source = {_feature_key(column): column for column in int_scalar_columns}
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        images = {
            camera: tfds.features.Image(
                shape=self._image_shapes[camera],
                dtype=np.uint8,
                encoding_format="jpeg",
            )
            for camera in self._cameras
        }
        step_features: dict[str, Any] = {
            "episode_index": np.int64,
            "source_episode_index": np.int64,
            "source_episode_id": tfds.features.Text(),
            "frame_index": np.int64,
            "timestamp": np.float32,
            "observation": {
                "state": tfds.features.Tensor(shape=(self._state_dim,), dtype=np.float32),
                "ee_pose": tfds.features.Tensor(shape=(self._state_dim,), dtype=np.float32),
                "images": images,
            },
            "action": tfds.features.Tensor(shape=(self._action_dim,), dtype=np.float32),
            "actions_cartesian": tfds.features.Tensor(
                shape=(self._action_chunk_length, self._action_dim),
                dtype=np.float32,
            ),
            "action_source_horizon": np.int64,
            "action_chunk_length": np.int64,
            "action_stride": np.int64,
            "quality": np.int64,
            "speed_bin": np.int64,
            "mistake": np.bool_,
            "success": np.bool_,
            "annotation_active_indices_json": tfds.features.Text(),
            "annotation_active_texts_json": tfds.features.Text(),
            "annotations_active_records_json": tfds.features.Text(),
            "discount": np.float32,
            "reward": np.float32,
            "is_first": np.bool_,
            "is_last": np.bool_,
            "is_terminal": np.bool_,
        }
        for column in TEXT_COLUMNS:
            step_features[column] = tfds.features.Text()
        if self._float_vector_columns:
            step_features["source_float_vectors"] = {
                _feature_key(column): tfds.features.Tensor(shape=shape, dtype=np.float32)
                for column, shape in self._float_vector_columns.items()
            }
        if self._int_scalar_columns:
            step_features["source_int_scalars"] = {
                _feature_key(column): np.int64 for column in self._int_scalar_columns
            }

        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "InfiData EgoVerse converted to semantic-preserving RLDS. "
                "Each generated dataset contains one robot_type + camera_set variant."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(step_features),
                    "episode_metadata": {
                        "episode_index": np.int64,
                        "source_episode_index": np.int64,
                        "source_episode_id": tfds.features.Text(),
                        "robot_schema_key": tfds.features.Text(),
                        "camera_set_json": tfds.features.Text(),
                        "image_fields_json": tfds.features.Text(),
                        "image_field_to_camera_json": tfds.features.Text(),
                        "parquet_path": tfds.features.Text(),
                        "source_metadata_path": tfds.features.Text(),
                        "num_frames": np.int64,
                        "fps": np.int64,
                        "duration_sec": np.float32,
                        "task": tfds.features.Text(),
                        "task_name": tfds.features.Text(),
                        "embodiment": tfds.features.Text(),
                        "robot_type": tfds.features.Text(),
                        "control_mode": tfds.features.Text(),
                        "quality": np.int64,
                        "speed_bin": np.int64,
                        "success": np.bool_,
                        "mistake": np.bool_,
                        "source_dataset": tfds.features.Text(),
                        "domain": tfds.features.Text(),
                        "state_action_schema_json": tfds.features.Text(),
                        "state_representation": tfds.features.Text(),
                        "action_representation": tfds.features.Text(),
                        "action_is_delta": np.bool_,
                        "cartesian_frame": tfds.features.Text(),
                        "cartesian_rotation": tfds.features.Text(),
                        "cartesian_layout": tfds.features.Text(),
                        "preserved_fields_json": tfds.features.Text(),
                        "source_float_vector_columns_json": tfds.features.Text(),
                        "source_int_scalar_columns_json": tfds.features.Text(),
                        "raw_episode_metadata_json": tfds.features.Text(),
                        "source_metadata_json": tfds.features.Text(),
                        "robots_json": tfds.features.Text(),
                        "tasks_json_path": tfds.features.Text(),
                        "stats_json_path": tfds.features.Text(),
                    },
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        splits = {"train": self._generate_examples("train", self._train_episodes)}
        if self._unseen_test_episodes:
            splits["unseen_test"] = self._generate_examples("unseen_test", self._unseen_test_episodes)
        if self._seen_test_episodes:
            splits["seen_test"] = self._generate_examples("seen_test", self._seen_test_episodes)
        return splits

    def _metadata_value(self, episode: InfiEpisode, key: str, fallback: Any) -> Any:
        return episode.raw_json.get(key, fallback)

    def _episode_metadata(self, episode: InfiEpisode, rows: pd.DataFrame) -> dict[str, Any]:
        first = rows.iloc[0]
        image_fields = tuple(f"images.{camera}" for camera in self._cameras)
        return {
            "episode_index": episode.episode_index,
            "source_episode_index": episode.source_episode_index,
            "source_episode_id": episode.source_episode_id,
            "robot_schema_key": self._schema_key,
            "camera_set_json": _json_dumps(list(self._cameras)),
            "image_fields_json": _json_dumps(list(image_fields)),
            "image_field_to_camera_json": _json_dumps({field: _camera_name(field) for field in image_fields}),
            "parquet_path": episode.parquet_path,
            "source_metadata_path": episode.source_metadata_path,
            "num_frames": int(self._metadata_value(episode, "num_frames", len(rows))),
            "fps": int(self._metadata_value(episode, "fps", self._robot_config.get("fps", 0))),
            "duration_sec": np.float32(self._metadata_value(episode, "duration_sec", 0.0)),
            "task": _as_text(first.get("task"), _as_text(self._metadata_value(episode, "task", ""))),
            "task_name": _as_text(first.get("task_name"), _as_text(self._metadata_value(episode, "task_name", ""))),
            "embodiment": _as_text(
                first.get("embodiment"), _as_text(self._metadata_value(episode, "embodiment", ""))
            ),
            "robot_type": _as_text(first.get("robot_type"), _as_text(self._metadata_value(episode, "robot_type", ""))),
            "control_mode": _as_text(
                first.get("control_mode"), _as_text(self._metadata_value(episode, "control_mode", ""))
            ),
            "quality": int(first.get("quality", self._metadata_value(episode, "quality", 0))),
            "speed_bin": int(first.get("speed_bin", self._metadata_value(episode, "speed_bin", 0))),
            "success": _as_bool(first.get("success", self._metadata_value(episode, "success", True))),
            "mistake": _as_bool(first.get("mistake", self._metadata_value(episode, "mistake", False))),
            "source_dataset": _as_text(
                first.get("source_dataset"), _as_text(self._metadata_value(episode, "source_dataset", ""))
            ),
            "domain": _as_text(first.get("domain"), _as_text(self._metadata_value(episode, "domain", ""))),
            "state_action_schema_json": _json_dumps(self._state_action_schema),
            "state_representation": self._state_action_schema["state_representation"],
            "action_representation": self._state_action_schema["action_representation"],
            "action_is_delta": bool(self._state_action_schema["action_is_delta"]),
            "cartesian_frame": _as_text(first.get("cartesian_frame"), self._state_action_schema["cartesian_frame"]),
            "cartesian_rotation": _as_text(
                first.get("cartesian_rotation"), self._state_action_schema["cartesian_rotation"]
            ),
            "cartesian_layout": _as_text(first.get("cartesian_layout"), ""),
            "preserved_fields_json": _json_dumps(self._metadata_value(episode, "preserved_fields", [])),
            "source_float_vector_columns_json": _json_dumps(self._float_key_to_source),
            "source_int_scalar_columns_json": _json_dumps(self._int_key_to_source),
            "raw_episode_metadata_json": _json_dumps(episode.raw_json),
            "source_metadata_json": _source_metadata_json(self._infidata_root, episode),
            "robots_json": _json_dumps(self._robots_json),
            "tasks_json_path": str(self._infidata_root / "meta" / "tasks.json"),
            "stats_json_path": str(self._infidata_root / "meta" / "stats.json"),
        }

    def _generate_examples(self, split_name: str, episodes: list[InfiEpisode]):
        total_frames = sum(episode.num_frames for episode in episodes)
        started = time.time()
        completed_frames = 0
        progress = tqdm.tqdm(
            total=len(episodes),
            desc=f"Converting EgoVerse {self._schema_key} {split_name} ({total_frames:,} frames)",
            unit="ep",
            dynamic_ncols=True,
        )

        image_columns = [f"images.{camera}" for camera in self._cameras]
        try:
            for episode in episodes:
                rows = pd.read_parquet(self._infidata_root / episode.parquet_path).reset_index(drop=True)
                if rows.empty:
                    progress.update(1)
                    continue
                length = len(rows)
                expected_columns = set(self._float_vector_columns) | set(self._int_scalar_columns)
                missing_columns = sorted(expected_columns - set(rows.columns))
                if missing_columns:
                    append_skip(
                        self._skip_log_path,
                        {
                            "dataset": "EgoVerse",
                            "split": split_name,
                            "episode_index": episode.episode_index,
                            "source_episode_index": episode.source_episode_index,
                            "schema_key": self._schema_key,
                            "parquet_path": episode.parquet_path,
                            "missing_columns": missing_columns,
                            "error": "missing_schema_columns",
                        },
                    )
                    print(
                        f"[skip] EgoVerse {split_name} episode={episode.episode_index}: "
                        f"missing columns {missing_columns}",
                        flush=True,
                    )
                    progress.update(1)
                    continue

                steps = []
                for i, row in rows.iterrows():
                    is_last = i == length - 1
                    step = {
                        "episode_index": int(row.get("episode_index", episode.episode_index)),
                        "source_episode_index": episode.source_episode_index,
                        "source_episode_id": _as_text(row.get("source_episode_id"), episode.source_episode_id),
                        "frame_index": int(row.get("frame_index", i)),
                        "timestamp": np.float32(row.get("timestamp", 0.0)),
                        "observation": {
                            "state": _as_float_array(row["observation.state"], (self._state_dim,)),
                            "ee_pose": _as_float_array(row["observations.state.ee_pose"], (self._state_dim,)),
                            "images": {
                                camera: row[column]
                                for camera, column in zip(self._cameras, image_columns, strict=True)
                            },
                        },
                        "action": _as_float_array(row["action"], (self._action_dim,)),
                        "actions_cartesian": _as_float_matrix(
                            row["actions_cartesian"],
                            (self._action_chunk_length, self._action_dim),
                        ),
                        "action_source_horizon": int(row.get("action_source_horizon", 0)),
                        "action_chunk_length": int(row.get("action_chunk_length", self._action_chunk_length)),
                        "action_stride": int(row.get("action_stride", 0)),
                        "quality": int(row.get("quality", 0)),
                        "speed_bin": int(row.get("speed_bin", 0)),
                        "mistake": _as_bool(row.get("mistake", False)),
                        "success": _as_bool(row.get("success", True)),
                        "annotation_active_indices_json": _annotation_indices_json(
                            row.get("annotations.active_indices")
                        ),
                        "annotation_active_texts_json": _annotation_texts_json(row.get("annotations.active_texts")),
                        "annotations_active_records_json": _as_text(
                            row.get("annotations.active_records_json"), "{}"
                        ),
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if is_last and _as_bool(row.get("success", True)) else 0.0),
                        "is_first": i == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                    }
                    for column in TEXT_COLUMNS:
                        step[column] = _as_text(row.get(column), "")
                    if self._float_vector_columns:
                        step["source_float_vectors"] = {
                            key: _as_float_array(row[source], self._float_vector_columns[source])
                            for key, source in self._float_key_to_source.items()
                        }
                    if self._int_scalar_columns:
                        step["source_int_scalars"] = {
                            key: int(row.get(source, 0)) for key, source in self._int_key_to_source.items()
                        }
                    steps.append(step)

                yield str(episode.episode_index), {
                    "steps": steps,
                    "episode_metadata": self._episode_metadata(episode, rows),
                }

                completed_frames += length
                elapsed = max(time.time() - started, 1e-6)
                frames_per_s = completed_frames / elapsed
                remaining_frames = max(total_frames - completed_frames, 0)
                progress.set_postfix(
                    frames_per_s=f"{frames_per_s:.1f}",
                    eta=f"{remaining_frames / max(frames_per_s, 1e-6) / 60:.1f}m",
                )
                progress.update(1)
        finally:
            progress.close()


def _parse_cameras(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _build_one_variant(
    *,
    infidata_root: Path,
    data_dir: Path,
    robots_json: dict[str, Any],
    all_episodes: list[InfiEpisode],
    variant: SchemaVariant,
    max_episodes: int | None,
    start_episode: int,
    unseen_test_ratio: float,
    seen_test_ratio: float,
    seed: int,
    overwrite: bool,
) -> None:
    selected_episodes = _select_episodes(
        all_episodes,
        schema_key=variant.schema_key,
        cameras=variant.cameras,
        start_episode=start_episode,
        max_episodes=max_episodes,
    )
    if not selected_episodes:
        raise ValueError(f"No episodes selected for schema={variant.schema_key}, cameras={variant.cameras}")

    robot_config = robots_json[variant.schema_key]
    canonical_episodes = _select_episodes(
        all_episodes,
        schema_key=variant.schema_key,
        cameras=variant.cameras,
        start_episode=0,
        max_episodes=None,
    )
    canonical_episode = canonical_episodes[0]
    sample_rows = _read_sample_rows(infidata_root, canonical_episode)
    image_shapes = _image_shapes(canonical_episode, robot_config)
    float_vector_columns, int_scalar_columns = _vector_columns(sample_rows)
    image_fields = tuple(f"images.{camera}" for camera in variant.cameras)
    state_action_schema = _state_action_schema(robot_config, image_fields)

    if overwrite:
        dataset_dir = data_dir / "ego_verse_infidata"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        skip_log_path = data_dir / "skipped_episodes.jsonl"
        if skip_log_path.exists():
            skip_log_path.unlink()
    else:
        skip_log_path = data_dir / "skipped_episodes.jsonl"

    expected_columns = set(float_vector_columns) | set(int_scalar_columns)
    valid_episodes: list[InfiEpisode] = []
    for episode in selected_episodes:
        parquet_size_bytes = _parquet_size_bytes(infidata_root, episode)
        if parquet_size_bytes > MAX_TFDS_EXAMPLE_PARQUET_BYTES:
            append_skip(
                skip_log_path,
                {
                    "dataset": "EgoVerse",
                    "split": "pre_split",
                    "episode_index": episode.episode_index,
                    "source_episode_index": episode.source_episode_index,
                    "schema_key": variant.schema_key,
                    "parquet_path": episode.parquet_path,
                    "parquet_size_bytes": parquet_size_bytes,
                    "max_parquet_size_bytes": MAX_TFDS_EXAMPLE_PARQUET_BYTES,
                    "error": "episode_too_large_for_tfds_example",
                },
            )
            continue
        missing_columns = sorted(expected_columns - _parquet_columns(infidata_root, episode))
        if missing_columns:
            append_skip(
                skip_log_path,
                {
                    "dataset": "EgoVerse",
                    "split": "pre_split",
                    "episode_index": episode.episode_index,
                    "source_episode_index": episode.source_episode_index,
                    "schema_key": variant.schema_key,
                    "parquet_path": episode.parquet_path,
                    "missing_columns": missing_columns,
                    "error": "missing_schema_columns",
                },
            )
            continue
        valid_episodes.append(episode)
    skipped_before_split = len(selected_episodes) - len(valid_episodes)
    selected_episodes = valid_episodes
    if not selected_episodes:
        print(
            f"No valid EgoVerse episodes remain for schema={variant.schema_key}, "
            f"cameras={','.join(variant.cameras)}; skipped={skipped_before_split}. "
            f"See {skip_log_path}."
        )
        return

    train_episodes, unseen_test_episodes, seen_test_episodes = _split_episodes(
        selected_episodes,
        unseen_test_ratio=unseen_test_ratio,
        seen_test_ratio=seen_test_ratio,
        seed=seed,
    )

    total_frames = sum(episode.num_frames for episode in selected_episodes)
    print(
        f"Selected {len(selected_episodes):,} EgoVerse episodes, {total_frames:,} frames, "
        f"schema={variant.schema_key}, cameras={','.join(variant.cameras)}, "
        f"state_dim={variant.state_dim}, action_dim={variant.action_dim}, "
        f"action_chunk_length={variant.action_chunk_length}, fps={variant.fps}."
    )
    print(
        f"Split: train={len(train_episodes):,}, unseen_test={len(unseen_test_episodes):,}, "
        f"seen_test={len(seen_test_episodes):,}, unseen_test_ratio={unseen_test_ratio:.3f}, "
        f"seen_test_ratio={seen_test_ratio:.3f}, seed={seed}."
    )
    print(f"Image shapes: {image_shapes}")
    print(f"Source float vector columns: {float_vector_columns}")
    print(f"Source int scalar columns: {list(int_scalar_columns)}")
    if skipped_before_split:
        print(f"Skipped {skipped_before_split:,} invalid or oversized episode(s) before split. See {skip_log_path}.")
    print(
        "Action semantics: actions_cartesian is a "
        f"{variant.action_chunk_length}x{variant.action_dim} absolute Cartesian pose chunk; "
        "action is actions_cartesian[0]; action_is_delta=False."
    )

    builder = EgoVerseInfidata(
        infidata_root=infidata_root,
        train_episodes=train_episodes,
        unseen_test_episodes=unseen_test_episodes,
        seen_test_episodes=seen_test_episodes,
        schema_key=variant.schema_key,
        cameras=variant.cameras,
        image_shapes=image_shapes,
        state_dim=variant.state_dim,
        action_dim=variant.action_dim,
        action_chunk_length=variant.action_chunk_length,
        robot_config=robot_config,
        robots_json=robots_json,
        float_vector_columns=float_vector_columns,
        int_scalar_columns=int_scalar_columns,
        state_action_schema=state_action_schema,
        skip_log_path=skip_log_path,
        data_dir=str(data_dir),
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            register_checksums=False,
            force_checksums_validation=False,
        ),
        download_dir=str(data_dir / "downloads"),
    )
    print(f"Built {builder.name}:{builder.version} at {builder.data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infidata-root", type=Path, default=Path("/mnt/workspace/InfiData/EgoVerse"))
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/workspace/RLDS/EgoVerse"))
    parser.add_argument("--schema-key", type=str, default=None)
    parser.add_argument("--camera-set", type=str, default=None, help="Comma-separated camera names, e.g. front_1.")
    parser.add_argument("--all-schemas", action="store_true", help="Build every robot_type + camera_set variant.")
    parser.add_argument("--list-schemas", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument(
        "--validation-ratio",
        "--unseen-test-ratio",
        dest="unseen_test_ratio",
        type=float,
        default=0.05,
        help="Held-out unseen test ratio sampled from all selected episodes.",
    )
    parser.add_argument(
        "--seen-test-ratio",
        type=float,
        default=0.05,
        help="Seen test ratio sampled from the train pool after removing unseen_test. Episodes stay in train.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic test sampling.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    robots_json = _load_robots_json(args.infidata_root)
    all_episodes = _load_all_episodes(args.infidata_root, robots_json)
    variants = _schema_variants(all_episodes, robots_json)

    if args.list_schemas:
        _print_schema_variants(variants)
        return

    if args.all_schemas and args.schema_key:
        raise ValueError("Use either --all-schemas or --schema-key, not both.")
    if not args.all_schemas and not args.schema_key:
        raise ValueError("Provide --schema-key, --all-schemas, or --list-schemas.")

    camera_set = _parse_cameras(args.camera_set)

    if args.all_schemas:
        selected_variants = variants
        if camera_set is not None:
            selected_variants = [variant for variant in selected_variants if variant.cameras == camera_set]
        if not selected_variants:
            raise ValueError(f"No variants matched camera_set={camera_set}.")
        for variant in selected_variants:
            _build_one_variant(
                infidata_root=args.infidata_root,
                data_dir=args.data_dir / variant.slug,
                robots_json=robots_json,
                all_episodes=all_episodes,
                variant=variant,
                max_episodes=args.max_episodes,
                start_episode=args.start_episode,
                unseen_test_ratio=args.unseen_test_ratio,
                seen_test_ratio=args.seen_test_ratio,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        return

    matching = [variant for variant in variants if variant.schema_key == args.schema_key]
    if camera_set is not None:
        matching = [variant for variant in matching if variant.cameras == camera_set]
    if not matching:
        raise ValueError(f"No EgoVerse variant matched schema_key={args.schema_key!r}, camera_set={camera_set!r}.")
    if len(matching) > 1:
        options = "\n".join(f"  --camera-set {','.join(variant.cameras)}" for variant in matching)
        raise ValueError(f"Schema {args.schema_key!r} has multiple camera sets; choose one:\n{options}")

    _build_one_variant(
        infidata_root=args.infidata_root,
        data_dir=args.data_dir,
        robots_json=robots_json,
        all_episodes=all_episodes,
        variant=matching[0],
        max_episodes=args.max_episodes,
        start_episode=args.start_episode,
        unseen_test_ratio=args.unseen_test_ratio,
        seen_test_ratio=args.seen_test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
