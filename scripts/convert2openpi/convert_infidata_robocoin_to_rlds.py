"""Convert InfiData RoboCOIN episodes to semantic-preserving RLDS/TFDS datasets.

RoboCOIN is heterogeneous: several robot types have multiple state/action
layouts, fps values, or camera sets. This converter therefore builds one RLDS
dataset per stable ``robot_schema_key + camera_set`` variant instead of forcing
all RoboCOIN episodes into a single schema.

Example smoke test:
    python scripts/convert2openpi/convert_infidata_robocoin_to_rlds.py \
        --infidata-root /mnt/workspace/InfiData/RoboCOIN \
        --schema-key discover_robotics_aitbot_mmk2_s36_a36_fps30 \
        --data-dir /mnt/workspace/tmp/robocoin_discover_s36_one \
        --max-episodes 1 \
        --overwrite

List convertible variants:
    python scripts/convert2openpi/convert_infidata_robocoin_to_rlds.py \
        --infidata-root /mnt/workspace/InfiData/RoboCOIN \
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

import av
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tqdm

from video_decode_utils import append_skip, ffmpeg_read_frames


TEXT_COLUMNS = (
    "task",
    "subtask",
    "robot_type",
    "control_mode",
    "source_dataset",
    "source_dataset_name",
    "domain",
    "subtask_annotation_status",
    "outcome_annotation_status",
)

OPTIONAL_INT_VECTOR_COLUMNS = {
    "source_subtask_annotation": (5,),
    "scene_annotation": (1,),
    "eef_direction_state": (2,),
    "eef_velocity_state": (2,),
    "eef_acc_mag_state": (2,),
    "eef_direction_action": (2,),
    "eef_velocity_action": (2,),
    "eef_acc_mag_action": (2,),
    "gripper_mode_state": (2,),
    "gripper_mode_action": (2,),
    "gripper_activity_state": (2,),
    "gripper_activity_action": (2,),
}

OPTIONAL_FLOAT_VECTOR_COLUMNS = {
    "eef_sim_pose_state": (12,),
    "eef_sim_pose_action": (12,),
    "gripper_open_scale_state": (2,),
    "gripper_open_scale_action": (2,),
}

ROBOCOIN_EEF_POSE_SCHEMA = {
    "field_names": [
        "left_eef_pos_x",
        "left_eef_pos_y",
        "left_eef_pos_z",
        "left_eef_ori_x",
        "left_eef_ori_y",
        "left_eef_ori_z",
        "right_eef_pos_x",
        "right_eef_pos_y",
        "right_eef_pos_z",
        "right_eef_ori_x",
        "right_eef_ori_y",
        "right_eef_ori_z",
    ],
    "coordinate_frame": "robocoin_simulation_space",
    "is_camera_frame": False,
    "can_transform_to_camera_frame_without_external_calibration": False,
    "description": (
        "RoboCOIN End-Effector Simulation Pose: left and right end-effector "
        "6D poses in RoboCOIN simulation space. The local dataset does not "
        "provide camera intrinsics/extrinsics or a documented transform from "
        "simulation space to camera coordinates."
    ),
}


@dataclasses.dataclass(frozen=True)
class InfiEpisode:
    episode_index: int
    source_episode_index: int
    parquet_path: str
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
    robot_type: str
    state_dim: int
    action_dim: int
    fps: int

    @property
    def slug(self) -> str:
        camera_slug = "_".join(self.cameras)
        return _slugify(f"{self.schema_key}_{camera_slug}")


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "robocoin_schema"


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
    return np.asarray(value, dtype=np.float32).reshape(shape)


def _as_int_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(shape)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_robots_json(infidata_root: Path) -> dict[str, Any]:
    robots = _load_json(infidata_root / "meta" / "robots.json")
    if not isinstance(robots, dict):
        raise ValueError(f"Expected object in {infidata_root / 'meta' / 'robots.json'}")
    return robots


def _source_to_schema(robots_json: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for schema_key, config in robots_json.items():
        if not isinstance(config, dict):
            continue
        for source_name in config.get("source_dataset_names", []):
            mapping[str(source_name)] = schema_key
    return mapping


def _camera_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, dict):
        return tuple(sorted(str(key) for key in raw))
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    return ()


def _load_all_episodes(infidata_root: Path, robots_json: dict[str, Any]) -> list[InfiEpisode]:
    source_schema = _source_to_schema(robots_json)
    episodes: list[InfiEpisode] = []
    with (infidata_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            source_name = _as_text(item.get("source_dataset_name"), "")
            schema_key = source_schema.get(source_name)
            if schema_key is None:
                raise KeyError(f"No robots.json schema for source_dataset_name={source_name!r}")
            episodes.append(
                InfiEpisode(
                    episode_index=int(item["episode_index"]),
                    source_episode_index=int(item.get("source_episode_index", -1)),
                    parquet_path=str(item["parquet_path"]),
                    raw_json=item,
                    num_frames=int(item.get("num_frames", 0)),
                    schema_key=schema_key,
                    cameras=_camera_tuple(item.get("video_paths", {})),
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
                robot_type=_as_text(config.get("robot_type"), ""),
                state_dim=int(config.get("state_dim", 0)),
                action_dim=int(config.get("action_dim", 0)),
                fps=int(config.get("fps", 0)),
            )
        )
    variants.sort(key=lambda item: (item.schema_key, item.cameras))
    return variants


def _print_schema_variants(variants: list[SchemaVariant]) -> None:
    print("Available RoboCOIN RLDS variants (one fixed RLDS schema per row):")
    for variant in variants:
        cameras = ",".join(variant.cameras) if variant.cameras else "<no_images>"
        print(
            f"- {variant.schema_key} | cameras={cameras} | episodes={variant.episode_count:,} | "
            f"frames={variant.frame_count:,} | robot_type={variant.robot_type} | "
            f"state_dim={variant.state_dim} | action_dim={variant.action_dim} | fps={variant.fps} | "
            f"slug={variant.slug}"
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


def _infer_columns(infidata_root: Path, episode: InfiEpisode) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rows = pd.read_parquet(infidata_root / episode.parquet_path)
    if rows.empty:
        raise ValueError(f"Empty episode parquet: {episode.parquet_path}")
    int_columns = tuple(name for name in OPTIONAL_INT_VECTOR_COLUMNS if name in rows.columns)
    float_columns = tuple(name for name in OPTIONAL_FLOAT_VECTOR_COLUMNS if name in rows.columns)
    return int_columns, float_columns


def _infer_image_shapes(infidata_root: Path, episode: InfiEpisode, cameras: tuple[str, ...]) -> dict[str, tuple[int, int, int]]:
    rows = pd.read_parquet(infidata_root / episode.parquet_path).reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"Empty episode parquet: {episode.parquet_path}")
    shapes: dict[str, tuple[int, int, int]] = {}
    for camera in cameras:
        rel_path = _as_text(rows[f"video.{camera}.path"].iloc[0])
        frame_index = int(rows[f"video.{camera}.frame_index"].iloc[0])
        frame = _read_frames_from_video(infidata_root / rel_path, [frame_index])[0]
        shapes[camera] = tuple(int(dim) for dim in frame.shape)
    return shapes


def _load_original_feature_schema(robocoin_root: Path | None, source_dataset_names: list[str]) -> dict[str, Any]:
    if robocoin_root is None:
        return {}
    for source_name in source_dataset_names:
        info_path = robocoin_root / source_name / "meta" / "info.json"
        if not info_path.exists():
            continue
        info = _load_json(info_path)
        if not isinstance(info, dict):
            continue
        features = info.get("features", {})
        if isinstance(features, dict):
            return {
                "source_dataset_name": source_name,
                "info_path": str(info_path),
                "state_feature": features.get("observation.state", {}),
                "action_feature": features.get("action", {}),
                "eef_sim_pose_state_feature": features.get("eef_sim_pose_state", {}),
                "eef_sim_pose_action_feature": features.get("eef_sim_pose_action", {}),
            }
    return {}


def _state_action_schema(
    schema_key: str,
    robot_config: dict[str, Any],
    original_feature_schema: dict[str, Any],
    int_columns: tuple[str, ...],
    float_columns: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_key": schema_key,
        "robot_type": robot_config.get("robot_type", ""),
        "control_mode": robot_config.get("control_mode", "joint"),
        "state_dim": int(robot_config.get("state_dim", 0)),
        "action_dim": int(robot_config.get("action_dim", 0)),
        "fps": int(robot_config.get("fps", 0)),
        "state_representation": "joint_state_vector",
        "action_representation": "joint_action_vector",
        "action_is_delta": "unknown",
        "action_description": (
            "RoboCOIN source joint action vector. The local metadata marks control_mode=joint, "
            "but does not globally define whether action is absolute or delta for all source datasets."
        ),
        "state_feature": original_feature_schema.get("state_feature", {}),
        "action_feature": original_feature_schema.get("action_feature", {}),
        "preserved_int_vector_columns": list(int_columns),
        "preserved_float_vector_columns": list(float_columns),
        "eef_pose_schema": ROBOCOIN_EEF_POSE_SCHEMA,
    }


def _enrich_robots_json(
    robots_json: dict[str, Any],
    *,
    schema_key: str,
    state_action_schema: dict[str, Any],
) -> dict[str, Any]:
    enriched = json.loads(json.dumps(robots_json, ensure_ascii=False))
    if schema_key in enriched and isinstance(enriched[schema_key], dict):
        enriched[schema_key].setdefault("state_action_schema", state_action_schema)
        enriched[schema_key]["state_representation"] = state_action_schema["state_representation"]
        enriched[schema_key]["action_representation"] = state_action_schema["action_representation"]
        enriched[schema_key]["action_is_delta"] = state_action_schema["action_is_delta"]
        enriched[schema_key]["eef_pose_coordinate_frame"] = ROBOCOIN_EEF_POSE_SCHEMA["coordinate_frame"]
        enriched[schema_key]["eef_pose_is_camera_frame"] = ROBOCOIN_EEF_POSE_SCHEMA["is_camera_frame"]
    return enriched


def _read_frames_from_video(path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    if not frame_indices:
        return []
    expected = list(int(index) for index in frame_indices)
    try:
        wanted = set(expected)
        max_index = max(wanted)
        frames_by_index: dict[int, np.ndarray] = {}

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for decoded_index, frame in enumerate(container.decode(stream)):
                if decoded_index in wanted:
                    frames_by_index[decoded_index] = frame.to_ndarray(format="rgb24")
                    if len(frames_by_index) == len(wanted):
                        break
                if decoded_index > max_index:
                    break

        missing = [index for index in expected if index not in frames_by_index]
        if missing:
            raise RuntimeError(
                f"Missing frame(s) {missing[:5]} from {path}; decoded {len(frames_by_index)} requested frames"
            )
        return [frames_by_index[index] for index in expected]
    except Exception as pyav_error:
        try:
            return ffmpeg_read_frames(path, expected)
        except Exception as ffmpeg_error:
            raise RuntimeError(
                f"Failed to read {len(expected)} frame(s) from {path}; "
                f"pyav={pyav_error!r}; ffmpeg={ffmpeg_error!r}"
            ) from ffmpeg_error


class RobocoinInfidata(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Semantic-preserving InfiData RoboCOIN RLDS split by robot schema and camera set."}

    def __init__(
        self,
        *,
        infidata_root: Path,
        train_episodes: list[InfiEpisode],
        unseen_test_episodes: list[InfiEpisode],
        seen_test_episodes: list[InfiEpisode],
        schema_key: str,
        cameras: tuple[str, ...],
        state_dim: int,
        action_dim: int,
        image_shapes: dict[str, tuple[int, int, int]],
        int_vector_columns: tuple[str, ...],
        float_vector_columns: tuple[str, ...],
        state_action_schema: dict[str, Any],
        robots_json: Any,
        original_feature_schema: dict[str, Any],
        skip_log_path: Path,
        **kwargs: Any,
    ):
        self._infidata_root = Path(infidata_root)
        self._train_episodes = train_episodes
        self._unseen_test_episodes = unseen_test_episodes
        self._seen_test_episodes = seen_test_episodes
        self._schema_key = schema_key
        self._cameras = cameras
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._image_shapes = image_shapes
        self._int_vector_columns = int_vector_columns
        self._float_vector_columns = float_vector_columns
        self._state_action_schema = state_action_schema
        self._robots_json = robots_json
        self._original_feature_schema = original_feature_schema
        self._skip_log_path = Path(skip_log_path)
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
            "frame_index": np.int64,
            "timestamp": np.float32,
            "observation": {
                "state": tfds.features.Tensor(shape=(self._state_dim,), dtype=np.float32),
                "images": images,
            },
            "action": tfds.features.Tensor(shape=(self._action_dim,), dtype=np.float32),
            "task": tfds.features.Text(),
            "subtask": tfds.features.Text(),
            "robot_type": tfds.features.Text(),
            "control_mode": tfds.features.Text(),
            "quality": np.int64,
            "speed_bin": np.int64,
            "mistake": np.bool_,
            "success": np.bool_,
            "source_dataset": tfds.features.Text(),
            "source_dataset_name": tfds.features.Text(),
            "domain": tfds.features.Text(),
            "subtask_annotation_status": tfds.features.Text(),
            "outcome_annotation_status": tfds.features.Text(),
            "video": {
                camera: {
                    "path": tfds.features.Text(),
                    "frame_index": np.int64,
                }
                for camera in self._cameras
            },
            "subgoal": {
                "real_future": {
                    camera: {
                        "path": tfds.features.Text(),
                        "frame_index": np.int64,
                    }
                    for camera in self._cameras
                }
            },
            "discount": np.float32,
            "reward": np.float32,
            "is_first": np.bool_,
            "is_last": np.bool_,
            "is_terminal": np.bool_,
        }
        for column in self._int_vector_columns:
            step_features[column] = tfds.features.Tensor(shape=OPTIONAL_INT_VECTOR_COLUMNS[column], dtype=np.int64)
        for column in self._float_vector_columns:
            step_features[column] = tfds.features.Tensor(shape=OPTIONAL_FLOAT_VECTOR_COLUMNS[column], dtype=np.float32)

        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "InfiData RoboCOIN converted to semantic-preserving RLDS. "
                "Each generated dataset contains one robot_schema_key + camera_set variant."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(step_features),
                    "episode_metadata": {
                        "episode_index": np.int64,
                        "source_episode_index": np.int64,
                        "robot_schema_key": tfds.features.Text(),
                        "camera_set_json": tfds.features.Text(),
                        "parquet_path": tfds.features.Text(),
                        "num_frames": np.int64,
                        "fps": np.int64,
                        "duration_sec": np.float32,
                        "task": tfds.features.Text(),
                        "robot_type": tfds.features.Text(),
                        "control_mode": tfds.features.Text(),
                        "quality": np.int64,
                        "speed_bin": np.int64,
                        "success": np.bool_,
                        "mistake": np.bool_,
                        "source_dataset": tfds.features.Text(),
                        "source_dataset_name": tfds.features.Text(),
                        "domain": tfds.features.Text(),
                        "video_paths_json": tfds.features.Text(),
                        "camera_mapping_json": tfds.features.Text(),
                        "state_action_schema_json": tfds.features.Text(),
                        "state_representation": tfds.features.Text(),
                        "action_representation": tfds.features.Text(),
                        "action_is_delta": tfds.features.Text(),
                        "eef_pose_schema_json": tfds.features.Text(),
                        "eef_pose_coordinate_frame": tfds.features.Text(),
                        "eef_pose_is_camera_frame": np.bool_,
                        "raw_episode_metadata_json": tfds.features.Text(),
                        "robots_json": tfds.features.Text(),
                        "original_feature_schema_json": tfds.features.Text(),
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
        return {
            "episode_index": episode.episode_index,
            "source_episode_index": episode.source_episode_index,
            "robot_schema_key": self._schema_key,
            "camera_set_json": _json_dumps(list(self._cameras)),
            "parquet_path": episode.parquet_path,
            "num_frames": int(self._metadata_value(episode, "num_frames", len(rows))),
            "fps": int(self._metadata_value(episode, "fps", self._state_action_schema.get("fps", 0))),
            "duration_sec": np.float32(self._metadata_value(episode, "duration_sec", 0.0)),
            "task": _as_text(first.get("task"), _as_text(self._metadata_value(episode, "task", ""))),
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
            "source_dataset_name": _as_text(
                first.get("source_dataset_name"), _as_text(self._metadata_value(episode, "source_dataset_name", ""))
            ),
            "domain": _as_text(first.get("domain"), _as_text(self._metadata_value(episode, "domain", ""))),
            "video_paths_json": _json_dumps(self._metadata_value(episode, "video_paths", {})),
            "camera_mapping_json": _json_dumps(self._metadata_value(episode, "camera_mapping", {})),
            "state_action_schema_json": _json_dumps(self._state_action_schema),
            "state_representation": self._state_action_schema["state_representation"],
            "action_representation": self._state_action_schema["action_representation"],
            "action_is_delta": _as_text(self._state_action_schema["action_is_delta"]),
            "eef_pose_schema_json": _json_dumps(ROBOCOIN_EEF_POSE_SCHEMA),
            "eef_pose_coordinate_frame": ROBOCOIN_EEF_POSE_SCHEMA["coordinate_frame"],
            "eef_pose_is_camera_frame": ROBOCOIN_EEF_POSE_SCHEMA["is_camera_frame"],
            "raw_episode_metadata_json": _json_dumps(episode.raw_json),
            "robots_json": _json_dumps(self._robots_json),
            "original_feature_schema_json": _json_dumps(self._original_feature_schema),
            "tasks_json_path": str(self._infidata_root / "meta" / "tasks.json"),
            "stats_json_path": str(self._infidata_root / "meta" / "stats.json"),
        }

    def _read_episode_images(self, rows: pd.DataFrame) -> dict[str, list[np.ndarray]]:
        clips: dict[str, list[np.ndarray]] = {}
        for camera in self._cameras:
            rel_path = _as_text(rows[f"video.{camera}.path"].iloc[0])
            path = self._infidata_root / rel_path
            frame_indices = [int(value) for value in rows[f"video.{camera}.frame_index"].tolist()]
            clips[camera] = _read_frames_from_video(path, frame_indices)
        return clips

    def _generate_examples(self, split_name: str, episodes: list[InfiEpisode]):
        total_frames = sum(episode.num_frames for episode in episodes)
        started = time.time()
        completed_frames = 0
        progress = tqdm.tqdm(
            total=len(episodes),
            desc=f"Converting RoboCOIN {self._schema_key} {split_name} ({total_frames:,} frames)",
            unit="ep",
            dynamic_ncols=True,
        )

        try:
            for episode in episodes:
                try:
                    rows = pd.read_parquet(self._infidata_root / episode.parquet_path).reset_index(drop=True)
                    if rows.empty:
                        progress.update(1)
                        continue
                    length = len(rows)
                    try:
                        clips = self._read_episode_images(rows)
                    except Exception as exc:
                        append_skip(
                            self._skip_log_path,
                            {
                                "dataset": "RoboCOIN",
                                "split": split_name,
                                "episode_index": episode.episode_index,
                                "source_episode_index": episode.source_episode_index,
                                "schema_key": self._schema_key,
                                "parquet_path": episode.parquet_path,
                                "error": repr(exc),
                            },
                        )
                        print(f"[skip] RoboCOIN {split_name} episode={episode.episode_index}: {exc}", flush=True)
                        progress.update(1)
                        continue

                    steps = []
                    for i, row in rows.iterrows():
                        is_last = i == length - 1
                        step = {
                            "episode_index": int(row.get("episode_index", episode.episode_index)),
                            "source_episode_index": int(
                                row.get("source_episode_index", episode.source_episode_index)
                            ),
                            "frame_index": int(row.get("frame_index", i)),
                            "timestamp": np.float32(row.get("timestamp", 0.0)),
                            "observation": {
                                "state": _as_float_array(row["observation.state"], (self._state_dim,)),
                                "images": {camera: clips[camera][i] for camera in self._cameras},
                            },
                            "action": _as_float_array(row["action"], (self._action_dim,)),
                            "quality": int(row.get("quality", 0)),
                            "speed_bin": int(row.get("speed_bin", 0)),
                            "mistake": _as_bool(row.get("mistake", False)),
                            "success": _as_bool(row.get("success", True)),
                            "video": {
                                camera: {
                                    "path": _as_text(row.get(f"video.{camera}.path"), ""),
                                    "frame_index": int(row.get(f"video.{camera}.frame_index", i)),
                                }
                                for camera in self._cameras
                            },
                            "subgoal": {
                                "real_future": {
                                    camera: {
                                        "path": _as_text(row.get(f"subgoal.real_future.{camera}.path"), ""),
                                        "frame_index": int(row.get(f"subgoal.real_future.{camera}.frame_index", i)),
                                    }
                                    for camera in self._cameras
                                }
                            },
                            "discount": np.float32(1.0),
                            "reward": np.float32(1.0 if is_last and _as_bool(row.get("success", True)) else 0.0),
                            "is_first": i == 0,
                            "is_last": is_last,
                            "is_terminal": is_last,
                        }
                        for column in TEXT_COLUMNS:
                            step[column] = _as_text(row.get(column), "")
                        for column in self._int_vector_columns:
                            step[column] = _as_int_array(row[column], OPTIONAL_INT_VECTOR_COLUMNS[column])
                        for column in self._float_vector_columns:
                            step[column] = _as_float_array(row[column], OPTIONAL_FLOAT_VECTOR_COLUMNS[column])
                        steps.append(step)

                    episode_metadata = self._episode_metadata(episode, rows)
                except Exception as exc:
                    append_skip(
                        self._skip_log_path,
                        {
                            "dataset": "RoboCOIN",
                            "split": split_name,
                            "episode_index": episode.episode_index,
                            "source_episode_index": episode.source_episode_index,
                            "schema_key": self._schema_key,
                            "parquet_path": episode.parquet_path,
                            "error": "episode_conversion_error",
                            "exception": repr(exc),
                        },
                    )
                    print(f"[skip] RoboCOIN {split_name} episode={episode.episode_index}: {exc}", flush=True)
                    progress.update(1)
                    continue

                yield str(episode.episode_index), {
                    "steps": steps,
                    "episode_metadata": episode_metadata,
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
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    return cameras


def _build_one_variant(
    *,
    infidata_root: Path,
    robocoin_root: Path | None,
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

    train_episodes, unseen_test_episodes, seen_test_episodes = _split_episodes(
        selected_episodes,
        unseen_test_ratio=unseen_test_ratio,
        seen_test_ratio=seen_test_ratio,
        seed=seed,
    )

    robot_config = robots_json[variant.schema_key]
    first_episode = selected_episodes[0]
    int_columns, float_columns = _infer_columns(infidata_root, first_episode)
    image_shapes = _infer_image_shapes(infidata_root, first_episode, variant.cameras)
    original_feature_schema = _load_original_feature_schema(
        robocoin_root,
        [str(name) for name in robot_config.get("source_dataset_names", [])],
    )
    state_action_schema = _state_action_schema(
        variant.schema_key,
        robot_config,
        original_feature_schema,
        int_columns,
        float_columns,
    )
    enriched_robots = _enrich_robots_json(
        robots_json,
        schema_key=variant.schema_key,
        state_action_schema=state_action_schema,
    )

    total_frames = sum(episode.num_frames for episode in selected_episodes)
    print(
        f"Selected {len(selected_episodes):,} RoboCOIN episodes, {total_frames:,} frames, "
        f"schema={variant.schema_key}, cameras={','.join(variant.cameras)}, "
        f"state_dim={variant.state_dim}, action_dim={variant.action_dim}, fps={variant.fps}."
    )
    print(
        f"Split: train={len(train_episodes):,}, unseen_test={len(unseen_test_episodes):,}, "
        f"seen_test={len(seen_test_episodes):,}, unseen_test_ratio={unseen_test_ratio:.3f}, "
        f"seen_test_ratio={seen_test_ratio:.3f}, seed={seed}."
    )
    print(f"Preserved int fields: {', '.join(int_columns) if int_columns else '<none>'}")
    print(f"Preserved float fields: {', '.join(float_columns) if float_columns else '<none>'}")
    print(f"Image shapes: {image_shapes}")

    if overwrite:
        dataset_dir = data_dir / "robocoin_infidata"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        skip_log_path = data_dir / "skipped_episodes.jsonl"
        if skip_log_path.exists():
            skip_log_path.unlink()
    else:
        skip_log_path = data_dir / "skipped_episodes.jsonl"

    builder = RobocoinInfidata(
        infidata_root=infidata_root,
        train_episodes=train_episodes,
        unseen_test_episodes=unseen_test_episodes,
        seen_test_episodes=seen_test_episodes,
        schema_key=variant.schema_key,
        cameras=variant.cameras,
        state_dim=variant.state_dim,
        action_dim=variant.action_dim,
        image_shapes=image_shapes,
        int_vector_columns=int_columns,
        float_vector_columns=float_columns,
        state_action_schema=state_action_schema,
        robots_json=enriched_robots,
        original_feature_schema=original_feature_schema,
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
    parser.add_argument("--infidata-root", type=Path, default=Path("/mnt/workspace/InfiData/RoboCOIN"))
    parser.add_argument(
        "--robocoin-root",
        type=Path,
        default=Path("/mnt/workspace/wudi/ELUBrain/RoboCOIN_data/RoboCOIN"),
        help="Optional original RoboCOIN LeRobot root used only to preserve feature name metadata.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/workspace/RLDS/RoboCOIN"))
    parser.add_argument("--schema-key", type=str, default=None)
    parser.add_argument("--camera-set", type=str, default=None, help="Comma-separated InfiData camera names.")
    parser.add_argument("--all-schemas", action="store_true", help="Build every robot_schema_key + camera_set variant.")
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

    robocoin_root = args.robocoin_root if args.robocoin_root and args.robocoin_root.exists() else None
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
                robocoin_root=robocoin_root,
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
        raise ValueError(f"No RoboCOIN variant matched schema_key={args.schema_key!r}, camera_set={camera_set!r}.")
    if len(matching) > 1:
        options = "\n".join(f"  --camera-set {','.join(variant.cameras)}" for variant in matching)
        raise ValueError(
            f"Schema {args.schema_key!r} has multiple camera sets; choose one explicitly:\n{options}"
        )

    _build_one_variant(
        infidata_root=args.infidata_root,
        robocoin_root=robocoin_root,
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
