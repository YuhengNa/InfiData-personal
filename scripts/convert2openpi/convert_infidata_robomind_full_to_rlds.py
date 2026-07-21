"""Convert full sharded InfiData RoboMIND data to semantic-preserving RLDS.

The full RoboMIND InfiData export is split across four shard directories and
contains multiple embodiments, state/action dimensions, camera sets, and action
semantics. This converter builds one RLDS dataset per stable schema variant:

    robot_type + source schema + embodiment + domain + state/action dims + cameras

Example:
    python scripts/convert2openpi/convert_infidata_robomind_full_to_rlds.py \
        --infidata-root /mnt/workspace/wudi/InfiData \
        --schema-key agilex_cobot_magic_agilex_dual_arm_h5_agilex_3rgb_real_s14_a14_fps30_cam_high_cam_left_wrist_cam_right_wrist \
        --data-dir /mnt/workspace/tmp/robomind_full_agilex_one \
        --max-episodes 1 \
        --overwrite

List variants:
    python scripts/convert2openpi/convert_infidata_robomind_full_to_rlds.py \
        --infidata-root /mnt/workspace/wudi/InfiData \
        --list-schemas
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tqdm

from video_decode_utils import append_skip, ffmpeg_read_frames


TEXT_COLUMNS = (
    "task",
    "subtask",
    "robot_type",
    "embodiment",
    "control_mode",
    "source_dataset",
    "domain",
    "subtask_annotation_status",
    "outcome_annotation_status",
)


@dataclasses.dataclass(frozen=True)
class ShardEpisode:
    shard_name: str
    shard_root: Path
    episode_index: int
    parquet_path: str
    raw_json: dict[str, Any]
    num_frames: int
    variant_key: str
    cameras: tuple[str, ...]

    @property
    def global_episode_key(self) -> str:
        return f"{self.shard_name}:{self.episode_index}"


@dataclasses.dataclass(frozen=True)
class SchemaVariant:
    schema_key: str
    robot_type: str
    source_schema: str
    embodiment: str
    domain: str
    control_mode: str
    state_dim: int
    action_dim: int
    fps: int
    cameras: tuple[str, ...]
    episode_count: int
    frame_count: int
    shard_counts: dict[str, int]
    success_counts: dict[str, int]
    quality_counts: dict[str, int]
    annotation_note: str


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "robomind_schema"


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


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _camera_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, dict):
        return tuple(sorted(str(key) for key in raw))
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    return ()


def _embodiment(value: Any) -> str:
    return _as_text(value, "none")


def _variant_key(item: dict[str, Any]) -> str:
    cameras = _camera_tuple(item.get("video_paths", {}))
    camera_slug = "_".join(cameras)
    return _slugify(
        "__".join(
            [
                _as_text(item.get("robot_type"), "unknown"),
                _as_text(item.get("schema"), "unknown_schema"),
                _embodiment(item.get("embodiment")),
                _as_text(item.get("domain"), "unknown_domain"),
                f"s{int(item.get('state_dim', 0))}_a{int(item.get('action_dim', 0))}_fps{int(item.get('fps', 0))}",
                camera_slug,
            ]
        )
    )


def _discover_shards(infidata_root: Path, shard_glob: str) -> list[Path]:
    shards = sorted(path for path in infidata_root.glob(shard_glob) if path.is_dir())
    if not shards:
        raise FileNotFoundError(f"No RoboMIND shard directories matched {infidata_root / shard_glob}")
    for shard in shards:
        if not (shard / "meta" / "episodes.jsonl").exists():
            raise FileNotFoundError(f"Missing episodes.jsonl under {shard}")
        if not (shard / "meta" / "robots.json").exists():
            raise FileNotFoundError(f"Missing robots.json under {shard}")
    return shards


def _load_all_episodes(shards: list[Path]) -> list[ShardEpisode]:
    episodes: list[ShardEpisode] = []
    for shard in shards:
        with (shard / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                episodes.append(
                    ShardEpisode(
                        shard_name=shard.name,
                        shard_root=shard,
                        episode_index=int(item["episode_index"]),
                        parquet_path=str(item["parquet_path"]),
                        raw_json=item,
                        num_frames=int(item.get("num_frames", 0)),
                        variant_key=_variant_key(item),
                        cameras=_camera_tuple(item.get("video_paths", {})),
                    )
                )
    return episodes


def _schema_variants(episodes: list[ShardEpisode]) -> list[SchemaVariant]:
    counts: Counter[str] = Counter()
    frames: Counter[str] = Counter()
    shard_counts: dict[str, Counter[str]] = defaultdict(Counter)
    success_counts: dict[str, Counter[str]] = defaultdict(Counter)
    quality_counts: dict[str, Counter[str]] = defaultdict(Counter)
    example: dict[str, dict[str, Any]] = {}
    notes: dict[str, Counter[str]] = defaultdict(Counter)

    for episode in episodes:
        key = episode.variant_key
        item = episode.raw_json
        counts[key] += 1
        frames[key] += episode.num_frames
        shard_counts[key][episode.shard_name] += 1
        success_counts[key][str(bool(item.get("success")))] += 1
        quality_counts[key][str(int(item.get("quality", -1)))] += 1
        notes[key][_as_text(item.get("annotation_note"), "")] += 1
        example.setdefault(key, item)

    variants: list[SchemaVariant] = []
    for key in sorted(counts):
        item = example[key]
        variants.append(
            SchemaVariant(
                schema_key=key,
                robot_type=_as_text(item.get("robot_type"), "unknown"),
                source_schema=_as_text(item.get("schema"), "unknown_schema"),
                embodiment=_embodiment(item.get("embodiment")),
                domain=_as_text(item.get("domain"), "unknown_domain"),
                control_mode=_as_text(item.get("control_mode"), "joint"),
                state_dim=int(item.get("state_dim", 0)),
                action_dim=int(item.get("action_dim", 0)),
                fps=int(item.get("fps", 0)),
                cameras=_camera_tuple(item.get("video_paths", {})),
                episode_count=counts[key],
                frame_count=frames[key],
                shard_counts=dict(shard_counts[key]),
                success_counts=dict(success_counts[key]),
                quality_counts=dict(quality_counts[key]),
                annotation_note=notes[key].most_common(1)[0][0] if notes[key] else "",
            )
        )
    variants.sort(key=lambda item: (-item.episode_count, item.schema_key))
    return variants


def _print_schema_variants(variants: list[SchemaVariant]) -> None:
    print("Available full RoboMIND RLDS variants:")
    for variant in variants:
        print(
            f"- {variant.schema_key} | episodes={variant.episode_count:,} | frames={variant.frame_count:,} | "
            f"robot_type={variant.robot_type} | schema={variant.source_schema} | "
            f"embodiment={variant.embodiment} | domain={variant.domain} | "
            f"state_dim={variant.state_dim} | action_dim={variant.action_dim} | fps={variant.fps} | "
            f"cameras={','.join(variant.cameras)} | shards={variant.shard_counts}"
        )


def _select_episodes(
    episodes: list[ShardEpisode],
    *,
    schema_key: str,
    max_episodes: int | None,
    start_episode: int,
) -> list[ShardEpisode]:
    selected = [
        episode
        for episode in episodes
        if episode.variant_key == schema_key and episode.episode_index >= start_episode
    ]
    selected.sort(key=lambda episode: (episode.shard_name, episode.episode_index))
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
    episodes: list[ShardEpisode],
    *,
    unseen_test_ratio: float,
    seen_test_ratio: float,
    seed: int,
) -> tuple[list[ShardEpisode], list[ShardEpisode], list[ShardEpisode]]:
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
    train.sort(key=lambda episode: (episode.shard_name, episode.episode_index))
    unseen_test.sort(key=lambda episode: (episode.shard_name, episode.episode_index))
    seen_test.sort(key=lambda episode: (episode.shard_name, episode.episode_index))
    return train, unseen_test, seen_test


def _read_episode_rows(episode: ShardEpisode) -> pd.DataFrame:
    return pd.read_parquet(episode.shard_root / episode.parquet_path).reset_index(drop=True)


def _read_frames_from_video(path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    return ffmpeg_read_frames(path, [int(index) for index in frame_indices])


def _ffprobe_image_shape(path: Path) -> tuple[int, int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe failed for {path}: {stderr}")
    info = json.loads(proc.stdout.decode("utf-8"))
    stream = info["streams"][0]
    return int(stream["height"]), int(stream["width"]), 3


def _infer_image_shapes(
    episodes: list[ShardEpisode],
    cameras: tuple[str, ...],
    skip_log_path: Path | None = None,
) -> dict[str, tuple[int, int, int]]:
    failures = []
    shape_counts: Counter[tuple[tuple[str, tuple[int, int, int]], ...]] = Counter()
    for episode in episodes:
        try:
            rows = _read_episode_rows(episode)
            if rows.empty:
                continue
            shapes: dict[str, tuple[int, int, int]] = {}
            for camera in cameras:
                rel_path = _as_text(rows[f"video.{camera}.path"].iloc[0])
                shapes[camera] = _ffprobe_image_shape(episode.shard_root / rel_path)
            shape_counts[tuple((camera, shapes[camera]) for camera in cameras)] += 1
        except Exception as exc:
            failures.append((episode, repr(exc)))
            if skip_log_path is not None:
                append_skip(
                    skip_log_path,
                    {
                        "dataset": "RoboMIND_full",
                        "split": "preflight",
                        "shard_name": episode.shard_name,
                        "episode_index": episode.episode_index,
                        "global_episode_key": episode.global_episode_key,
                        "schema_key": episode.variant_key,
                        "parquet_path": episode.parquet_path,
                        "error": "image_shape_inference_error",
                        "exception": repr(exc),
                    },
                )
    if not shape_counts:
        raise RuntimeError(f"Could not infer image shapes for cameras={cameras}; failures={failures[:5]}")

    most_common_shapes, most_common_count = shape_counts.most_common(1)[0]
    if len(shape_counts) > 1:
        print(
            "Image shape candidates: "
            + ", ".join(
                f"{dict(signature)} -> {count:,} episode(s)"
                for signature, count in shape_counts.most_common()
            )
            + f"; using most common from {most_common_count:,}/{sum(shape_counts.values()):,} episode(s)."
        )
    return dict(most_common_shapes)


def _load_robots_by_shard(shards: list[Path]) -> dict[str, Any]:
    return {shard.name: _load_json(shard / "meta" / "robots.json") for shard in shards}


def _tasks_paths_by_shard(shards: list[Path]) -> dict[str, str]:
    return {shard.name: str(shard / "meta" / "tasks.json") for shard in shards}


def _stats_paths_by_shard(shards: list[Path]) -> dict[str, str]:
    return {shard.name: str(shard / "meta" / "stats.json") for shard in shards if (shard / "meta" / "stats.json").exists()}


def _state_action_schema(variant: SchemaVariant) -> dict[str, Any]:
    if variant.source_schema in {"master_puppet_joint_position", "agilex_dual_arm"}:
        action_representation = "absolute_master_joint_position"
        action_is_delta: bool | str = False
        action_description = "RoboMIND master joint positions aligned to observation frames."
    elif variant.source_schema in {"simulation_franka_joint_position", "tiangong_joint_position"}:
        action_representation = "next_joint_state"
        action_is_delta = False
        action_description = "RoboMIND next joint state aligned to RGB frame count."
    else:
        action_representation = "joint_action_vector"
        action_is_delta = "unknown"
        action_description = "RoboMIND source joint action vector; exact temporal semantics are schema-dependent."

    return {
        "schema_key": variant.schema_key,
        "robot_type": variant.robot_type,
        "source_schema": variant.source_schema,
        "embodiment": variant.embodiment,
        "domain": variant.domain,
        "control_mode": variant.control_mode,
        "state_dim": variant.state_dim,
        "action_dim": variant.action_dim,
        "fps": variant.fps,
        "state_representation": "joint_position_state",
        "action_representation": action_representation,
        "action_is_delta": action_is_delta,
        "state_description": "RoboMIND joint-position state, schema-dependent.",
        "action_description": action_description,
        "annotation_note": variant.annotation_note,
        "cameras": list(variant.cameras),
    }


class RobomindFullInfidata(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Full sharded RoboMIND InfiData converted to semantic-preserving RLDS by schema."}

    def __init__(
        self,
        *,
        train_episodes: list[ShardEpisode],
        unseen_test_episodes: list[ShardEpisode],
        seen_test_episodes: list[ShardEpisode],
        variant: SchemaVariant,
        image_shapes: dict[str, tuple[int, int, int]],
        state_action_schema: dict[str, Any],
        robots_json_by_shard: dict[str, Any],
        tasks_paths_by_shard: dict[str, str],
        stats_paths_by_shard: dict[str, str],
        skip_log_path: Path,
        **kwargs: Any,
    ):
        self._train_episodes = train_episodes
        self._unseen_test_episodes = unseen_test_episodes
        self._seen_test_episodes = seen_test_episodes
        self._variant = variant
        self._cameras = variant.cameras
        self._image_shapes = image_shapes
        self._state_action_schema = state_action_schema
        self._robots_json_by_shard = robots_json_by_shard
        self._tasks_paths_by_shard = tasks_paths_by_shard
        self._stats_paths_by_shard = stats_paths_by_shard
        self._skip_log_path = Path(skip_log_path)
        self._video_shape_cache: dict[str, tuple[int, int, int]] = {}
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
        return tfds.core.DatasetInfo(
            builder=self,
            description="Full RoboMIND InfiData converted to semantic-preserving RLDS.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "episode_index": np.int64,
                            "shard_name": tfds.features.Text(),
                            "global_episode_key": tfds.features.Text(),
                            "frame_index": np.int64,
                            "timestamp": np.float32,
                            "observation": {
                                "state": tfds.features.Tensor(shape=(self._variant.state_dim,), dtype=np.float32),
                                "images": images,
                            },
                            "action": tfds.features.Tensor(shape=(self._variant.action_dim,), dtype=np.float32),
                            "task": tfds.features.Text(),
                            "subtask": tfds.features.Text(),
                            "robot_type": tfds.features.Text(),
                            "embodiment": tfds.features.Text(),
                            "control_mode": tfds.features.Text(),
                            "quality": np.int64,
                            "speed_bin": np.int64,
                            "mistake": np.bool_,
                            "success": np.bool_,
                            "source_dataset": tfds.features.Text(),
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
                    ),
                    "episode_metadata": {
                        "episode_index": np.int64,
                        "shard_name": tfds.features.Text(),
                        "global_episode_key": tfds.features.Text(),
                        "robot_schema_key": tfds.features.Text(),
                        "source_schema": tfds.features.Text(),
                        "embodiment": tfds.features.Text(),
                        "camera_set_json": tfds.features.Text(),
                        "parquet_path": tfds.features.Text(),
                        "source_file": tfds.features.Text(),
                        "source_annotation_id": tfds.features.Text(),
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
                        "domain": tfds.features.Text(),
                        "video_paths_json": tfds.features.Text(),
                        "camera_mapping_json": tfds.features.Text(),
                        "state_action_schema_json": tfds.features.Text(),
                        "state_representation": tfds.features.Text(),
                        "action_representation": tfds.features.Text(),
                        "action_is_delta": tfds.features.Text(),
                        "annotation_note": tfds.features.Text(),
                        "raw_episode_metadata_json": tfds.features.Text(),
                        "robots_json_by_shard": tfds.features.Text(),
                        "tasks_paths_by_shard": tfds.features.Text(),
                        "stats_paths_by_shard": tfds.features.Text(),
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

    def _episode_metadata(self, episode: ShardEpisode, rows: pd.DataFrame) -> dict[str, Any]:
        first = rows.iloc[0]
        raw = episode.raw_json
        return {
            "episode_index": episode.episode_index,
            "shard_name": episode.shard_name,
            "global_episode_key": episode.global_episode_key,
            "robot_schema_key": self._variant.schema_key,
            "source_schema": self._variant.source_schema,
            "embodiment": self._variant.embodiment,
            "camera_set_json": _json_dumps(list(self._cameras)),
            "parquet_path": episode.parquet_path,
            "source_file": _as_text(raw.get("source_file"), ""),
            "source_annotation_id": _as_text(raw.get("source_annotation_id"), ""),
            "num_frames": int(raw.get("num_frames", len(rows))),
            "fps": int(raw.get("fps", self._variant.fps)),
            "duration_sec": np.float32(raw.get("duration_sec", 0.0)),
            "task": _as_text(first.get("task"), _as_text(raw.get("task"), "")),
            "robot_type": _as_text(first.get("robot_type"), _as_text(raw.get("robot_type"), "")),
            "control_mode": _as_text(first.get("control_mode"), _as_text(raw.get("control_mode"), "")),
            "quality": int(first.get("quality", raw.get("quality", 0))),
            "speed_bin": int(first.get("speed_bin", raw.get("speed_bin", 0))),
            "success": _as_bool(first.get("success", raw.get("success", True))),
            "mistake": _as_bool(first.get("mistake", raw.get("mistake", False))),
            "source_dataset": _as_text(first.get("source_dataset"), _as_text(raw.get("source_dataset"), "")),
            "domain": _as_text(first.get("domain"), _as_text(raw.get("domain"), "")),
            "video_paths_json": _json_dumps(raw.get("video_paths", {})),
            "camera_mapping_json": _json_dumps(raw.get("camera_mapping", {})),
            "state_action_schema_json": _json_dumps(self._state_action_schema),
            "state_representation": self._state_action_schema["state_representation"],
            "action_representation": self._state_action_schema["action_representation"],
            "action_is_delta": _as_text(self._state_action_schema["action_is_delta"]),
            "annotation_note": _as_text(raw.get("annotation_note"), ""),
            "raw_episode_metadata_json": _json_dumps(raw),
            "robots_json_by_shard": _json_dumps(self._robots_json_by_shard),
            "tasks_paths_by_shard": _json_dumps(self._tasks_paths_by_shard),
            "stats_paths_by_shard": _json_dumps(self._stats_paths_by_shard),
        }

    def _read_episode_images(self, episode: ShardEpisode, rows: pd.DataFrame) -> dict[str, list[np.ndarray]]:
        clips: dict[str, list[np.ndarray]] = {}
        for camera in self._cameras:
            rel_path = _as_text(rows[f"video.{camera}.path"].iloc[0])
            frame_indices = [int(value) for value in rows[f"video.{camera}.frame_index"].tolist()]
            clips[camera] = _read_frames_from_video(episode.shard_root / rel_path, frame_indices)
        return clips

    def _validate_episode_image_shapes(self, episode: ShardEpisode, rows: pd.DataFrame) -> None:
        mismatches = {}
        for camera in self._cameras:
            rel_path = _as_text(rows[f"video.{camera}.path"].iloc[0])
            video_path = episode.shard_root / rel_path
            cache_key = str(video_path)
            actual_shape = self._video_shape_cache.get(cache_key)
            if actual_shape is None:
                actual_shape = _ffprobe_image_shape(video_path)
                self._video_shape_cache[cache_key] = actual_shape
            expected_shape = self._image_shapes[camera]
            if actual_shape != expected_shape:
                mismatches[camera] = {
                    "path": str(video_path),
                    "expected_shape": expected_shape,
                    "actual_shape": actual_shape,
                }
        if mismatches:
            raise ValueError(f"incompatible_image_shape: {mismatches}")

    def _generate_examples(self, split_name: str, episodes: list[ShardEpisode]):
        total_frames = sum(episode.num_frames for episode in episodes)
        started = time.time()
        completed_frames = 0
        progress = tqdm.tqdm(
            total=len(episodes),
            desc=f"Converting RoboMIND full {self._variant.schema_key} {split_name} ({total_frames:,} frames)",
            unit="ep",
            dynamic_ncols=True,
        )

        try:
            for episode in episodes:
                try:
                    rows = _read_episode_rows(episode)
                    if rows.empty:
                        progress.update(1)
                        continue
                    length = len(rows)
                    try:
                        self._validate_episode_image_shapes(episode, rows)
                    except Exception as exc:
                        append_skip(
                            self._skip_log_path,
                            {
                                "dataset": "RoboMIND_full",
                                "split": split_name,
                                "shard_name": episode.shard_name,
                                "episode_index": episode.episode_index,
                                "global_episode_key": episode.global_episode_key,
                                "schema_key": self._variant.schema_key,
                                "parquet_path": episode.parquet_path,
                                "error": "incompatible_image_shape",
                                "exception": repr(exc),
                            },
                        )
                        print(
                            f"[skip] RoboMIND_full {split_name} episode={episode.global_episode_key}: {exc}",
                            flush=True,
                        )
                        progress.update(1)
                        continue
                    try:
                        clips = self._read_episode_images(episode, rows)
                    except Exception as exc:
                        append_skip(
                            self._skip_log_path,
                            {
                                "dataset": "RoboMIND_full",
                                "split": split_name,
                                "shard_name": episode.shard_name,
                                "episode_index": episode.episode_index,
                                "global_episode_key": episode.global_episode_key,
                                "schema_key": self._variant.schema_key,
                                "parquet_path": episode.parquet_path,
                                "error": "video_decode_error",
                                "exception": repr(exc),
                            },
                        )
                        print(
                            f"[skip] RoboMIND_full {split_name} episode={episode.global_episode_key}: {exc}",
                            flush=True,
                        )
                        progress.update(1)
                        continue

                    steps = []
                    for i, row in rows.iterrows():
                        is_last = i == length - 1
                        step = {
                            "episode_index": int(row.get("episode_index", episode.episode_index)),
                            "shard_name": episode.shard_name,
                            "global_episode_key": episode.global_episode_key,
                            "frame_index": int(row.get("frame_index", i)),
                            "timestamp": np.float32(row.get("timestamp", 0.0)),
                            "observation": {
                                "state": _as_float_array(row["observation.state"], (self._variant.state_dim,)),
                                "images": {camera: clips[camera][i] for camera in self._cameras},
                            },
                            "action": _as_float_array(row["action"], (self._variant.action_dim,)),
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
                        steps.append(step)

                    episode_metadata = self._episode_metadata(episode, rows)
                except Exception as exc:
                    append_skip(
                        self._skip_log_path,
                        {
                            "dataset": "RoboMIND_full",
                            "split": split_name,
                            "shard_name": episode.shard_name,
                            "episode_index": episode.episode_index,
                            "global_episode_key": episode.global_episode_key,
                            "schema_key": self._variant.schema_key,
                            "parquet_path": episode.parquet_path,
                            "error": "episode_conversion_error",
                            "exception": repr(exc),
                        },
                    )
                    print(f"[skip] RoboMIND_full {split_name} episode={episode.global_episode_key}: {exc}", flush=True)
                    progress.update(1)
                    continue

                yield episode.global_episode_key, {
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


def _build_one_variant(
    *,
    shards: list[Path],
    all_episodes: list[ShardEpisode],
    variant: SchemaVariant,
    data_dir: Path,
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
        max_episodes=max_episodes,
        start_episode=start_episode,
    )
    if not selected_episodes:
        raise ValueError(f"No episodes selected for schema={variant.schema_key}")

    if overwrite:
        dataset_dir = data_dir / "robomind_full_infidata"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        skip_log_path = data_dir / "skipped_episodes.jsonl"
        if skip_log_path.exists():
            skip_log_path.unlink()
    else:
        skip_log_path = data_dir / "skipped_episodes.jsonl"

    image_shapes = _infer_image_shapes(selected_episodes, variant.cameras, skip_log_path=skip_log_path)
    state_action_schema = _state_action_schema(variant)
    train_episodes, unseen_test_episodes, seen_test_episodes = _split_episodes(
        selected_episodes,
        unseen_test_ratio=unseen_test_ratio,
        seen_test_ratio=seen_test_ratio,
        seed=seed,
    )

    total_frames = sum(episode.num_frames for episode in selected_episodes)
    print(
        f"Selected {len(selected_episodes):,} full RoboMIND episodes, {total_frames:,} frames, "
        f"schema={variant.schema_key}, cameras={','.join(variant.cameras)}, "
        f"state_dim={variant.state_dim}, action_dim={variant.action_dim}, fps={variant.fps}."
    )
    print(f"Shard counts: {variant.shard_counts}; success={variant.success_counts}; quality={variant.quality_counts}")
    print(
        f"Split: train={len(train_episodes):,}, unseen_test={len(unseen_test_episodes):,}, "
        f"seen_test={len(seen_test_episodes):,}, unseen_test_ratio={unseen_test_ratio:.3f}, "
        f"seen_test_ratio={seen_test_ratio:.3f}, seed={seed}."
    )
    print(f"Image shapes: {image_shapes}")
    print(
        f"Action semantics: {state_action_schema['action_representation']}; "
        f"action_is_delta={state_action_schema['action_is_delta']}."
    )

    builder = RobomindFullInfidata(
        train_episodes=train_episodes,
        unseen_test_episodes=unseen_test_episodes,
        seen_test_episodes=seen_test_episodes,
        variant=variant,
        image_shapes=image_shapes,
        state_action_schema=state_action_schema,
        robots_json_by_shard=_load_robots_by_shard(shards),
        tasks_paths_by_shard=_tasks_paths_by_shard(shards),
        stats_paths_by_shard=_stats_paths_by_shard(shards),
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
    parser.add_argument("--infidata-root", type=Path, default=Path("/mnt/workspace/wudi/InfiData"))
    parser.add_argument("--shard-glob", type=str, default="RoboMIND_shard_*")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/workspace/RLDS/RoboMIND_full"))
    parser.add_argument("--schema-key", type=str, default=None)
    parser.add_argument("--all-schemas", action="store_true")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    shards = _discover_shards(args.infidata_root, args.shard_glob)
    all_episodes = _load_all_episodes(shards)
    variants = _schema_variants(all_episodes)

    if args.list_schemas:
        _print_schema_variants(variants)
        return

    if args.all_schemas and args.schema_key:
        raise ValueError("Use either --all-schemas or --schema-key, not both.")
    if not args.all_schemas and not args.schema_key:
        raise ValueError("Provide --schema-key, --all-schemas, or --list-schemas.")

    variants_by_key = {variant.schema_key: variant for variant in variants}
    if args.all_schemas:
        for variant in variants:
            _build_one_variant(
                shards=shards,
                all_episodes=all_episodes,
                variant=variant,
                data_dir=args.data_dir / f"{variant.schema_key}__episodes_{variant.episode_count}",
                max_episodes=args.max_episodes,
                start_episode=args.start_episode,
                unseen_test_ratio=args.unseen_test_ratio,
                seen_test_ratio=args.seen_test_ratio,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        return

    variant = variants_by_key.get(args.schema_key)
    if variant is None:
        raise ValueError(f"Unknown schema_key={args.schema_key!r}. Run --list-schemas to inspect available variants.")

    _build_one_variant(
        shards=shards,
        all_episodes=all_episodes,
        variant=variant,
        data_dir=args.data_dir,
        max_episodes=args.max_episodes,
        start_episode=args.start_episode,
        unseen_test_ratio=args.unseen_test_ratio,
        seen_test_ratio=args.seen_test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
