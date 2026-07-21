"""Convert InfiData AgiBotWorld-Beta ModelScope data to semantic-preserving RLDS.

The converter follows the same scan-then-convert flow used by the RoboMIND and
RoboCOIN converters:

1. Scan meta/episodes.jsonl and group episodes by a stable schema key.
2. Select one schema with --schema-key, or build every schema with --all-schemas.
3. Split selected episodes into train, unseen_test, and seen_test.
4. Decode videos and write one RLDS repository per schema.

Example:
    python scripts/convert2openpi/convert_infidata_agibot_to_rlds.py \
        --infidata-root /mnt/workspace/InfiData/AgiBotWorld-Beta-ModelScope-500h \
        --schema-key agibot_world_robot_agibot_world_beta_mobile_dual_arm_joint_absolute_position_real_s20_a20_fps30_cam_high_cam_left_wrist_cam_right_wrist \
        --data-dir /mnt/workspace/tmp/agibot_one \
        --max-episodes 1 \
        --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
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
    "action_type",
    "quality_source",
    "success_source",
    "mistake_source",
    "speed_bin_source",
    "source_dataset",
    "domain",
    "source_task_name",
    "init_scene_text",
    "subtask_skill",
    "subtask_annotation_status",
    "outcome_annotation_status",
)


@dataclasses.dataclass(frozen=True)
class AgibotEpisode:
    episode_index: int
    parquet_path: str
    raw_json: dict[str, Any]
    num_frames: int
    variant_key: str
    cameras: tuple[str, ...]

    @property
    def global_episode_key(self) -> str:
        return f"agibot:{self.episode_index}"


@dataclasses.dataclass(frozen=True)
class SchemaVariant:
    schema_key: str
    robot_type: str
    embodiment: str
    control_mode: str
    action_type: str
    domain: str
    state_dim: int
    action_dim: int
    fps: int
    cameras: tuple[str, ...]
    state_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    state_dims: tuple[str, ...]
    action_dims: tuple[str, ...]
    episode_count: int
    frame_count: int
    task_count: int
    speed_bin_counts: dict[str, int]
    success_counts: dict[str, int]
    quality_counts: dict[str, int]
    annotation_note: str


class SequentialVideoFrameReader:
    """Sequential ffmpeg rawvideo reader for long episodes."""

    def __init__(self, path: Path, shape: tuple[int, int, int]):
        self.path = path
        self.shape = shape
        self.frame_size = int(np.prod(shape))
        self.current_index = -1
        self.proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "SequentialVideoFrameReader":
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(self.path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        # Do not pipe stderr here: verbose decoder errors can fill the pipe and
        # deadlock while the caller is blocked on stdout.read(frame_size).
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def read(self, target_index: int) -> np.ndarray:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("reader is not open")
        target_index = int(target_index)
        if target_index < self.current_index:
            raise ValueError(f"Cannot seek backwards in {self.path}: {target_index} < {self.current_index}")

        frame: np.ndarray | None = None
        while self.current_index < target_index:
            data = self.proc.stdout.read(self.frame_size)
            if len(data) != self.frame_size:
                raise RuntimeError(
                    f"Short ffmpeg read for {self.path} at frame {self.current_index + 1}; "
                    f"got {len(data)} bytes, expected {self.frame_size}."
                )
            self.current_index += 1
            if self.current_index == target_index:
                frame = np.frombuffer(data, dtype=np.uint8).reshape(self.shape).copy()
        if frame is None:
            raise RuntimeError(f"No frame decoded for {self.path} index={target_index}")
        return frame


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "agibot_schema"


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


def _tuple_text(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(str(item) for item in raw)


def _variant_key(item: dict[str, Any]) -> str:
    cameras = _camera_tuple(item.get("video_paths", {}))
    return _slugify(
        "__".join(
            [
                _as_text(item.get("robot_type"), "unknown_robot"),
                _as_text(item.get("embodiment"), "unknown_embodiment"),
                _as_text(item.get("control_mode"), "unknown_control"),
                _as_text(item.get("action_type"), "unknown_action_type"),
                _as_text(item.get("domain"), "unknown_domain"),
                f"s{int(item.get('state_dim', 0))}_a{int(item.get('action_dim', 0))}_fps{int(item.get('fps', 0))}",
                "_".join(cameras),
            ]
        )
    )


def _load_all_episodes(infidata_root: Path) -> list[AgibotEpisode]:
    episodes_path = infidata_root / "meta" / "episodes.jsonl"
    robots_path = infidata_root / "meta" / "robots.json"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}")
    if not robots_path.exists():
        raise FileNotFoundError(f"Missing {robots_path}")

    episodes: list[AgibotEpisode] = []
    with episodes_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            episodes.append(
                AgibotEpisode(
                    episode_index=int(item["episode_index"]),
                    parquet_path=str(item["parquet_path"]),
                    raw_json=item,
                    num_frames=int(item.get("num_frames", 0)),
                    variant_key=_variant_key(item),
                    cameras=_camera_tuple(item.get("video_paths", {})),
                )
            )
    return episodes


def _schema_variants(episodes: list[AgibotEpisode]) -> list[SchemaVariant]:
    counts: Counter[str] = Counter()
    frames: Counter[str] = Counter()
    task_ids: dict[str, set[int]] = defaultdict(set)
    speed_bins: dict[str, Counter[str]] = defaultdict(Counter)
    successes: dict[str, Counter[str]] = defaultdict(Counter)
    qualities: dict[str, Counter[str]] = defaultdict(Counter)
    notes: dict[str, Counter[str]] = defaultdict(Counter)
    example: dict[str, dict[str, Any]] = {}

    for episode in episodes:
        item = episode.raw_json
        key = episode.variant_key
        counts[key] += 1
        frames[key] += episode.num_frames
        if item.get("source_task_id") is not None:
            task_ids[key].add(int(item["source_task_id"]))
        speed_bins[key][str(int(item.get("speed_bin", -1)))] += 1
        successes[key][str(bool(item.get("success")))] += 1
        qualities[key][str(int(item.get("quality", -1)))] += 1
        notes[key][_as_text(item.get("annotation_note"), "")] += 1
        example.setdefault(key, item)

    variants: list[SchemaVariant] = []
    for key in sorted(counts):
        item = example[key]
        variants.append(
            SchemaVariant(
                schema_key=key,
                robot_type=_as_text(item.get("robot_type"), "unknown_robot"),
                embodiment=_as_text(item.get("embodiment"), "unknown_embodiment"),
                control_mode=_as_text(item.get("control_mode"), "unknown_control"),
                action_type=_as_text(item.get("action_type"), "unknown_action_type"),
                domain=_as_text(item.get("domain"), "unknown_domain"),
                state_dim=int(item.get("state_dim", 0)),
                action_dim=int(item.get("action_dim", 0)),
                fps=int(item.get("fps", 0)),
                cameras=_camera_tuple(item.get("video_paths", {})),
                state_fields=_tuple_text(item.get("state_fields")),
                action_fields=_tuple_text(item.get("action_fields")),
                state_dims=_tuple_text(item.get("state_dims")),
                action_dims=_tuple_text(item.get("action_dims")),
                episode_count=counts[key],
                frame_count=frames[key],
                task_count=len(task_ids[key]),
                speed_bin_counts=dict(speed_bins[key]),
                success_counts=dict(successes[key]),
                quality_counts=dict(qualities[key]),
                annotation_note=notes[key].most_common(1)[0][0] if notes[key] else "",
            )
        )
    variants.sort(key=lambda item: (-item.episode_count, item.schema_key))
    return variants


def _print_schema_variants(variants: list[SchemaVariant]) -> None:
    print("Available AgiBot RLDS variants:")
    for variant in variants:
        print(
            f"- {variant.schema_key} | episodes={variant.episode_count:,} | frames={variant.frame_count:,} | "
            f"tasks={variant.task_count:,} | robot_type={variant.robot_type} | embodiment={variant.embodiment} | "
            f"control_mode={variant.control_mode} | action_type={variant.action_type} | domain={variant.domain} | "
            f"state_dim={variant.state_dim} | action_dim={variant.action_dim} | fps={variant.fps} | "
            f"cameras={','.join(variant.cameras)}"
        )


def _select_episodes(
    episodes: list[AgibotEpisode],
    *,
    schema_key: str,
    max_episodes: int | None,
    start_episode: int,
) -> list[AgibotEpisode]:
    selected = [
        episode
        for episode in episodes
        if episode.variant_key == schema_key and episode.episode_index >= start_episode
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
    episodes: list[AgibotEpisode],
    *,
    unseen_test_ratio: float,
    seen_test_ratio: float,
    seed: int,
) -> tuple[list[AgibotEpisode], list[AgibotEpisode], list[AgibotEpisode]]:
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


def _read_episode_rows(infidata_root: Path, episode: AgibotEpisode) -> pd.DataFrame:
    return pd.read_parquet(infidata_root / episode.parquet_path).reset_index(drop=True)


def _read_frames_from_video(path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    return ffmpeg_read_frames(path, [int(index) for index in frame_indices])


def _ffprobe_frame_count(path: Path) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe failed for {path}: {stderr}")
    info = json.loads(proc.stdout.decode("utf-8"))
    raw = info.get("streams", [{}])[0].get("nb_frames")
    if raw in (None, "N/A"):
        return None
    return int(raw)


def _infer_image_shapes(
    infidata_root: Path,
    episodes: list[AgibotEpisode],
    cameras: tuple[str, ...],
    skip_log_path: Path | None = None,
) -> dict[str, tuple[int, int, int]]:
    failures = []
    for episode in episodes:
        try:
            rows = _read_episode_rows(infidata_root, episode)
            if rows.empty:
                continue
            shapes: dict[str, tuple[int, int, int]] = {}
            for camera in cameras:
                rel_path = _as_text(rows[f"video.{camera}.path"].iloc[0])
                frame_index = int(rows[f"video.{camera}.frame_index"].iloc[0])
                frame = _read_frames_from_video(infidata_root / rel_path, [frame_index])[0]
                shapes[camera] = tuple(int(dim) for dim in frame.shape)
            return shapes
        except Exception as exc:
            failures.append((episode, repr(exc)))
            if skip_log_path is not None:
                append_skip(
                    skip_log_path,
                    {
                        "dataset": "AgiBotWorld-Beta-ModelScope-500h",
                        "split": "preflight",
                        "episode_index": episode.episode_index,
                        "global_episode_key": episode.global_episode_key,
                        "schema_key": episode.variant_key,
                        "parquet_path": episode.parquet_path,
                        "error": "image_shape_inference_error",
                        "exception": repr(exc),
                    },
                )
    raise RuntimeError(f"Could not infer image shapes for cameras={cameras}; failures={failures[:5]}")


def _load_stats_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        stats = json.load(f)
    return {key: value for key, value in stats.items() if key != "episodes"}


def _state_action_schema(variant: SchemaVariant) -> dict[str, Any]:
    if variant.action_type == "absolute_position":
        action_is_delta: bool | str = False
        action_representation = "agibot_absolute_joint_effector_head_waist_position"
        action_description = "AgiBot absolute target positions built from joint, effector, head, and waist position fields."
    else:
        action_is_delta = "unknown"
        action_representation = "agibot_action_vector"
        action_description = "AgiBot source action vector; exact temporal semantics are action_type-dependent."

    return {
        "schema_key": variant.schema_key,
        "robot_type": variant.robot_type,
        "embodiment": variant.embodiment,
        "domain": variant.domain,
        "control_mode": variant.control_mode,
        "action_type": variant.action_type,
        "state_dim": variant.state_dim,
        "action_dim": variant.action_dim,
        "fps": variant.fps,
        "state_fields": list(variant.state_fields),
        "action_fields": list(variant.action_fields),
        "state_dims": list(variant.state_dims),
        "action_dims": list(variant.action_dims),
        "state_representation": "agibot_joint_effector_head_waist_position",
        "action_representation": action_representation,
        "action_is_delta": action_is_delta,
        "state_description": "AgiBot state built from joint, effector, head, and waist position fields.",
        "action_description": action_description,
        "annotation_note": variant.annotation_note,
        "cameras": list(variant.cameras),
    }


class AgibotInfidata(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "AgiBotWorld-Beta ModelScope 500h InfiData converted to semantic-preserving RLDS."}

    def __init__(
        self,
        *,
        infidata_root: Path,
        train_episodes: list[AgibotEpisode],
        unseen_test_episodes: list[AgibotEpisode],
        seen_test_episodes: list[AgibotEpisode],
        variant: SchemaVariant,
        image_shapes: dict[str, tuple[int, int, int]],
        state_action_schema: dict[str, Any],
        robots_json: dict[str, Any],
        tasks_json: dict[str, Any],
        stats_summary_json: dict[str, Any],
        skip_log_path: Path,
        **kwargs: Any,
    ):
        self._infidata_root = Path(infidata_root)
        self._train_episodes = train_episodes
        self._unseen_test_episodes = unseen_test_episodes
        self._seen_test_episodes = seen_test_episodes
        self._variant = variant
        self._cameras = variant.cameras
        self._image_shapes = image_shapes
        self._state_action_schema = state_action_schema
        self._robots_json = robots_json
        self._tasks_json = tasks_json
        self._stats_summary_json = stats_summary_json
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
        return tfds.core.DatasetInfo(
            builder=self,
            description="AgiBotWorld-Beta ModelScope InfiData converted to semantic-preserving RLDS.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "episode_index": np.int64,
                            "global_episode_key": tfds.features.Text(),
                            "frame_index": np.int64,
                            "timestamp": np.float32,
                            "observation": {
                                "state": tfds.features.Tensor(shape=(self._variant.state_dim,), dtype=np.float32),
                                "images": images,
                            },
                            "action": tfds.features.Tensor(shape=(self._variant.action_dim,), dtype=np.float32),
                            "source_task_id": np.int64,
                            "source_episode_id": np.int64,
                            "source_action_config_index": np.float32,
                            "quality": np.int64,
                            "speed_bin": np.int64,
                            "mistake": np.bool_,
                            "success": np.bool_,
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
                            **{column: tfds.features.Text() for column in TEXT_COLUMNS},
                        }
                    ),
                    "episode_metadata": {
                        "episode_index": np.int64,
                        "global_episode_key": tfds.features.Text(),
                        "robot_schema_key": tfds.features.Text(),
                        "embodiment": tfds.features.Text(),
                        "camera_set_json": tfds.features.Text(),
                        "parquet_path": tfds.features.Text(),
                        "source_task_id": np.int64,
                        "source_episode_id": np.int64,
                        "source_proprio_file": tfds.features.Text(),
                        "source_observation_dir": tfds.features.Text(),
                        "source_parameters_dir": tfds.features.Text(),
                        "source_init_scene_text": tfds.features.Text(),
                        "source_label_info_json": tfds.features.Text(),
                        "num_frames": np.int64,
                        "fps": np.int64,
                        "duration_sec": np.float32,
                        "task": tfds.features.Text(),
                        "robot_type": tfds.features.Text(),
                        "control_mode": tfds.features.Text(),
                        "action_type": tfds.features.Text(),
                        "quality": np.int64,
                        "speed_bin": np.int64,
                        "success": np.bool_,
                        "mistake": np.bool_,
                        "quality_source": tfds.features.Text(),
                        "success_source": tfds.features.Text(),
                        "mistake_source": tfds.features.Text(),
                        "speed_bin_source": tfds.features.Text(),
                        "source_dataset": tfds.features.Text(),
                        "domain": tfds.features.Text(),
                        "video_paths_json": tfds.features.Text(),
                        "camera_mapping_json": tfds.features.Text(),
                        "state_action_schema_json": tfds.features.Text(),
                        "state_fields_json": tfds.features.Text(),
                        "action_fields_json": tfds.features.Text(),
                        "state_dims_json": tfds.features.Text(),
                        "action_dims_json": tfds.features.Text(),
                        "state_representation": tfds.features.Text(),
                        "action_representation": tfds.features.Text(),
                        "action_is_delta": tfds.features.Text(),
                        "annotation_note": tfds.features.Text(),
                        "raw_episode_metadata_json": tfds.features.Text(),
                        "robots_json": tfds.features.Text(),
                        "tasks_json": tfds.features.Text(),
                        "stats_summary_json": tfds.features.Text(),
                        "stats_path": tfds.features.Text(),
                        "segments_path": tfds.features.Text(),
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

    def _episode_metadata(self, episode: AgibotEpisode, rows: pd.DataFrame) -> dict[str, Any]:
        first = rows.iloc[0]
        raw = episode.raw_json
        return {
            "episode_index": episode.episode_index,
            "global_episode_key": episode.global_episode_key,
            "robot_schema_key": self._variant.schema_key,
            "embodiment": self._variant.embodiment,
            "camera_set_json": _json_dumps(list(self._cameras)),
            "parquet_path": episode.parquet_path,
            "source_task_id": int(raw.get("source_task_id", first.get("source_task_id", -1))),
            "source_episode_id": int(raw.get("source_episode_id", first.get("source_episode_id", -1))),
            "source_proprio_file": _as_text(raw.get("source_proprio_file"), ""),
            "source_observation_dir": _as_text(raw.get("source_observation_dir"), ""),
            "source_parameters_dir": _as_text(raw.get("source_parameters_dir"), ""),
            "source_init_scene_text": _as_text(raw.get("source_init_scene_text"), ""),
            "source_label_info_json": _json_dumps(raw.get("source_label_info", {})),
            "num_frames": int(raw.get("num_frames", len(rows))),
            "fps": int(raw.get("fps", self._variant.fps)),
            "duration_sec": np.float32(raw.get("duration_sec", 0.0)),
            "task": _as_text(first.get("task"), _as_text(raw.get("task"), "")),
            "robot_type": _as_text(first.get("robot_type"), _as_text(raw.get("robot_type"), "")),
            "control_mode": _as_text(first.get("control_mode"), _as_text(raw.get("control_mode"), "")),
            "action_type": _as_text(first.get("action_type"), _as_text(raw.get("action_type"), "")),
            "quality": int(first.get("quality", raw.get("quality", 0))),
            "speed_bin": int(first.get("speed_bin", raw.get("speed_bin", 0))),
            "success": _as_bool(first.get("success", raw.get("success", True))),
            "mistake": _as_bool(first.get("mistake", raw.get("mistake", False))),
            "quality_source": _as_text(first.get("quality_source"), _as_text(raw.get("quality_source"), "")),
            "success_source": _as_text(first.get("success_source"), _as_text(raw.get("success_source"), "")),
            "mistake_source": _as_text(first.get("mistake_source"), _as_text(raw.get("mistake_source"), "")),
            "speed_bin_source": _as_text(first.get("speed_bin_source"), _as_text(raw.get("speed_bin_source"), "")),
            "source_dataset": _as_text(first.get("source_dataset"), _as_text(raw.get("source_dataset"), "")),
            "domain": _as_text(first.get("domain"), _as_text(raw.get("domain"), "")),
            "video_paths_json": _json_dumps(raw.get("video_paths", {})),
            "camera_mapping_json": _json_dumps(raw.get("camera_mapping", {})),
            "state_action_schema_json": _json_dumps(self._state_action_schema),
            "state_fields_json": _json_dumps(raw.get("state_fields", [])),
            "action_fields_json": _json_dumps(raw.get("action_fields", [])),
            "state_dims_json": _json_dumps(raw.get("state_dims", [])),
            "action_dims_json": _json_dumps(raw.get("action_dims", [])),
            "state_representation": self._state_action_schema["state_representation"],
            "action_representation": self._state_action_schema["action_representation"],
            "action_is_delta": _as_text(self._state_action_schema["action_is_delta"]),
            "annotation_note": _as_text(raw.get("annotation_note"), ""),
            "raw_episode_metadata_json": _json_dumps(raw),
            "robots_json": _json_dumps(self._robots_json),
            "tasks_json": _json_dumps(self._tasks_json),
            "stats_summary_json": _json_dumps(self._stats_summary_json),
            "stats_path": str(self._infidata_root / "meta" / "stats.json"),
            "segments_path": str(self._infidata_root / "meta" / "segments.jsonl"),
        }

    def _validate_video_frame_counts(self, episode: AgibotEpisode, rows: pd.DataFrame) -> None:
        for camera in self._cameras:
            rel_path = _as_text(rows[f"video.{camera}.path"].iloc[0])
            path = self._infidata_root / rel_path
            if not path.exists():
                raise FileNotFoundError(path)
            max_frame_index = int(rows[f"video.{camera}.frame_index"].max())
            frame_count = _ffprobe_frame_count(path)
            if frame_count is not None and max_frame_index >= frame_count:
                raise RuntimeError(
                    f"Video {path} has {frame_count} frames but episode requests frame {max_frame_index}"
                )

    def _stream_episode_steps(self, episode: AgibotEpisode, rows: pd.DataFrame):
        length = len(rows)
        with contextlib.ExitStack() as stack:
            readers = {}
            first = rows.iloc[0]
            for camera in self._cameras:
                rel_path = _as_text(first.get(f"video.{camera}.path"), "")
                readers[camera] = stack.enter_context(
                    SequentialVideoFrameReader(self._infidata_root / rel_path, self._image_shapes[camera])
                )

            for i, row in rows.iterrows():
                is_last = i == length - 1
                try:
                    images = {
                        camera: readers[camera].read(int(row.get(f"video.{camera}.frame_index", i)))
                        for camera in self._cameras
                    }
                except Exception as exc:
                    append_skip(
                        self._skip_log_path,
                        {
                            "dataset": "AgiBotWorld-Beta-ModelScope-500h",
                            "split": "nested_step_stream",
                            "episode_index": episode.episode_index,
                            "global_episode_key": episode.global_episode_key,
                            "schema_key": self._variant.schema_key,
                            "parquet_path": episode.parquet_path,
                            "frame_index": int(row.get("frame_index", i)),
                            "error": "video_stream_decode_error",
                            "exception": repr(exc),
                        },
                    )
                    raise

                step = {
                    "episode_index": int(row.get("episode_index", episode.episode_index)),
                    "global_episode_key": episode.global_episode_key,
                    "frame_index": int(row.get("frame_index", i)),
                    "timestamp": np.float32(row.get("timestamp", 0.0)),
                    "observation": {
                        "state": _as_float_array(row["observation.state"], (self._variant.state_dim,)),
                        "images": images,
                    },
                    "action": _as_float_array(row["action"], (self._variant.action_dim,)),
                    "source_task_id": int(row.get("source_task_id", -1)),
                    "source_episode_id": int(row.get("source_episode_id", -1)),
                    "source_action_config_index": np.float32(row.get("source_action_config_index", np.nan)),
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
                yield step

    def _generate_examples(self, split_name: str, episodes: list[AgibotEpisode]):
        total_frames = sum(episode.num_frames for episode in episodes)
        started = time.time()
        scheduled_frames = 0
        progress = tqdm.tqdm(
            total=len(episodes),
            desc=f"Converting AgiBot {self._variant.schema_key} {split_name} ({total_frames:,} frames)",
            unit="ep",
            dynamic_ncols=True,
        )

        try:
            for episode in episodes:
                try:
                    rows = _read_episode_rows(self._infidata_root, episode)
                    if rows.empty:
                        progress.update(1)
                        continue
                    self._validate_video_frame_counts(episode, rows)
                    episode_metadata = self._episode_metadata(episode, rows)
                    steps = self._stream_episode_steps(episode, rows)
                except Exception as exc:
                    append_skip(
                        self._skip_log_path,
                        {
                            "dataset": "AgiBotWorld-Beta-ModelScope-500h",
                            "split": split_name,
                            "episode_index": episode.episode_index,
                            "global_episode_key": episode.global_episode_key,
                            "schema_key": self._variant.schema_key,
                            "parquet_path": episode.parquet_path,
                            "error": "episode_preflight_error",
                            "exception": repr(exc),
                        },
                    )
                    print(f"[skip] AgiBot {split_name} episode={episode.global_episode_key}: {exc}", flush=True)
                    progress.update(1)
                    continue

                yield episode.global_episode_key, {
                    "steps": steps,
                    "episode_metadata": episode_metadata,
                }

                scheduled_frames += len(rows)
                elapsed = max(time.time() - started, 1e-6)
                frames_per_s = scheduled_frames / elapsed
                remaining_frames = max(total_frames - scheduled_frames, 0)
                progress.set_postfix(
                    frames_per_s=f"{frames_per_s:.1f}",
                    eta=f"{remaining_frames / max(frames_per_s, 1e-6) / 60:.1f}m",
                )
                progress.update(1)
        finally:
            progress.close()


def _build_one_variant(
    *,
    infidata_root: Path,
    all_episodes: list[AgibotEpisode],
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

    skip_log_path = data_dir / "skipped_episodes.jsonl"
    if overwrite:
        dataset_dir = data_dir / "agibot_infidata"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        if skip_log_path.exists():
            skip_log_path.unlink()

    image_shapes = _infer_image_shapes(infidata_root, selected_episodes, variant.cameras, skip_log_path=skip_log_path)
    state_action_schema = _state_action_schema(variant)
    train_episodes, unseen_test_episodes, seen_test_episodes = _split_episodes(
        selected_episodes,
        unseen_test_ratio=unseen_test_ratio,
        seen_test_ratio=seen_test_ratio,
        seed=seed,
    )

    total_frames = sum(episode.num_frames for episode in selected_episodes)
    print(
        f"Selected {len(selected_episodes):,} AgiBot episodes, {total_frames:,} frames, "
        f"schema={variant.schema_key}, cameras={','.join(variant.cameras)}, "
        f"state_dim={variant.state_dim}, action_dim={variant.action_dim}, fps={variant.fps}."
    )
    print(
        f"Tasks={variant.task_count:,}; speed_bins={variant.speed_bin_counts}; "
        f"success={variant.success_counts}; quality={variant.quality_counts}"
    )
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

    builder = AgibotInfidata(
        infidata_root=infidata_root,
        train_episodes=train_episodes,
        unseen_test_episodes=unseen_test_episodes,
        seen_test_episodes=seen_test_episodes,
        variant=variant,
        image_shapes=image_shapes,
        state_action_schema=state_action_schema,
        robots_json=_load_json(infidata_root / "meta" / "robots.json") or {},
        tasks_json=_load_json(infidata_root / "meta" / "tasks.json") or {},
        stats_summary_json=_load_stats_summary(infidata_root / "meta" / "stats.json"),
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
    parser.add_argument(
        "--infidata-root",
        type=Path,
        default=Path("/mnt/workspace/InfiData/AgiBotWorld-Beta-ModelScope-500h"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/workspace/RLDS/AgiBot"))
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

    all_episodes = _load_all_episodes(args.infidata_root)
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
                infidata_root=args.infidata_root,
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
        infidata_root=args.infidata_root,
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
