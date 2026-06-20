"""Convert InfiData DROID episodes to a semantic-preserving RLDS/TFDS dataset.

This converter keeps DROID's InfiData schema semantics instead of reshaping
episodes into openpi's historical DROID loader schema. Training code should add
a dataset adapter that maps these fields into the model input/output contract.

Images are decoded from the original LeRobot DROID root because the InfiData
video links may point at a stale absolute mount and frame indices are packed
within shared videos.

Example smoke test:
    python scripts/convert2openpi/convert_infidata_droid_to_rlds.py \
        --infidata-root /mnt/workspace/InfiData/DROID \
        --lerobot-root /mnt/workspace/wudi/ELUBrain/DROID \
        --data-dir /mnt/workspace/tmp/droid_infidata_rlds_test \
        --max-episodes 3 \
        --overwrite
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tqdm

from video_decode_utils import append_skip, ffmpeg_read_clip


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SOURCE_CAMERA_BY_INFI = {
    "cam_high": "observation.images.exterior_1_left",
    "cam_left_wrist": "observation.images.wrist_left",
    "cam_right_wrist": "observation.images.exterior_2_left",
}
TEXT_COLUMNS = (
    "task",
    "subtask",
    "robot_type",
    "control_mode",
    "source_dataset",
    "domain",
    "subtask_annotation_status",
    "task_category",
    "building",
    "collector_id",
    "date",
)
DROID_STATE_ACTION_SCHEMA = {
    "control_mode": "joint",
    "state_representation": "absolute_joint_position_with_gripper",
    "state_description": "Franka 7D joint positions followed by 1D gripper position",
    "action_representation": "absolute_joint_position_with_gripper",
    "action_description": "Franka 7D joint position target followed by 1D gripper position target",
    "action_is_delta": False,
    "state_layout": ["franka_joint_position[7]", "gripper_position[1]"],
    "action_layout": ["franka_joint_position_target[7]", "gripper_position_target[1]"],
}


@dataclasses.dataclass(frozen=True)
class InfiEpisode:
    episode_index: int
    source_episode_index: int
    parquet_path: str
    raw_json: dict[str, Any]
    num_frames: int


@dataclasses.dataclass(frozen=True)
class OriginalEpisode:
    episode_index: int
    length: int
    local_video_starts: dict[str, int]
    video_chunk_indices: dict[str, int]
    video_file_indices: dict[str, int]


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


def _enrich_robots_json(robots_json: Any) -> Any:
    if not isinstance(robots_json, dict):
        return robots_json
    enriched = json.loads(json.dumps(robots_json, ensure_ascii=False))
    for robot in enriched.values():
        if not isinstance(robot, dict):
            continue
        if robot.get("robot_type") != "franka":
            continue
        robot.setdefault("state_action_schema", DROID_STATE_ACTION_SCHEMA)
        robot.setdefault("state_representation", DROID_STATE_ACTION_SCHEMA["state_representation"])
        robot.setdefault("action_representation", DROID_STATE_ACTION_SCHEMA["action_representation"])
        robot.setdefault("action_is_delta", DROID_STATE_ACTION_SCHEMA["action_is_delta"])
        robot["state_description"] = DROID_STATE_ACTION_SCHEMA["state_description"]
        robot["action_description"] = DROID_STATE_ACTION_SCHEMA["action_description"]
    return enriched


def _load_infidata_episodes(infidata_root: Path, max_episodes: int | None, start_episode: int) -> list[InfiEpisode]:
    episodes: list[InfiEpisode] = []
    with (infidata_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            episode_index = int(item["episode_index"])
            if episode_index < start_episode:
                continue
            episodes.append(
                InfiEpisode(
                    episode_index=episode_index,
                    source_episode_index=int(item["source_episode_index"]),
                    parquet_path=str(item["parquet_path"]),
                    raw_json=item,
                    num_frames=int(item.get("num_frames", 0)),
                )
            )
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
    return episodes


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


def _infer_vector_dim(infidata_root: Path, episode: InfiEpisode, column: str) -> int:
    rows = pd.read_parquet(infidata_root / episode.parquet_path, columns=[column])
    if rows.empty:
        raise ValueError(f"Empty episode parquet: {episode.parquet_path}")
    return int(np.asarray(rows[column].iloc[0], dtype=np.float32).reshape(-1).shape[0])


def _load_original_episode_index(lerobot_root: Path) -> dict[int, OriginalEpisode]:
    episode_files = sorted((lerobot_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No LeRobot episode metadata found under {lerobot_root / 'meta' / 'episodes'}")

    frames = [pd.read_parquet(path) for path in episode_files]
    meta = pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)

    for infi_camera, source_camera in SOURCE_CAMERA_BY_INFI.items():
        chunk_col = f"videos/{source_camera}/chunk_index"
        file_col = f"videos/{source_camera}/file_index"
        local_col = f"local_video_start/{infi_camera}"
        meta[local_col] = meta.groupby([chunk_col, file_col], sort=False)["length"].cumsum() - meta["length"]

    index: dict[int, OriginalEpisode] = {}
    for _, row in meta.iterrows():
        episode_index = int(row["episode_index"])
        index[episode_index] = OriginalEpisode(
            episode_index=episode_index,
            length=int(row["length"]),
            local_video_starts={camera: int(row[f"local_video_start/{camera}"]) for camera in CAMERAS},
            video_chunk_indices={
                camera: int(row[f"videos/{source_camera}/chunk_index"])
                for camera, source_camera in SOURCE_CAMERA_BY_INFI.items()
            },
            video_file_indices={
                camera: int(row[f"videos/{source_camera}/file_index"])
                for camera, source_camera in SOURCE_CAMERA_BY_INFI.items()
            },
        )
    return index


class VideoReaderCache:
    def __init__(self, max_open: int = 6):
        self._max_open = max_open
        self._cache: OrderedDict[Path, tuple[av.container.InputContainer, av.video.stream.VideoStream]] = OrderedDict()

    def close(self) -> None:
        for container, _ in self._cache.values():
            container.close()
        self._cache.clear()

    def _get(self, path: Path) -> tuple[av.container.InputContainer, av.video.stream.VideoStream]:
        if path in self._cache:
            container, stream = self._cache.pop(path)
            self._cache[path] = (container, stream)
            return container, stream

        if len(self._cache) >= self._max_open:
            _, (old_container, _) = self._cache.popitem(last=False)
            old_container.close()

        container = av.open(str(path))
        stream = container.streams.video[0]
        self._cache[path] = (container, stream)
        return container, stream

    def read_clip(self, path: Path, start_frame: int, length: int) -> list[np.ndarray]:
        try:
            container, stream = self._get(path)
            fps = float(stream.average_rate)
            target_pts = int(start_frame / fps / float(stream.time_base))
            container.seek(target_pts, stream=stream, backward=True)

            frames: list[np.ndarray] = []
            for frame in container.decode(stream):
                frame_index = round(float(frame.pts * stream.time_base * stream.average_rate))
                if frame_index < start_frame:
                    continue
                if frame_index >= start_frame + length:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == length:
                    break

            if len(frames) != length:
                raise RuntimeError(f"Expected {length} frames from {path} @ {start_frame}, got {len(frames)}")
            return frames
        except Exception as pyav_error:
            cached = self._cache.pop(path, None)
            if cached is not None:
                cached[0].close()
            try:
                return ffmpeg_read_clip(path, start_frame, length)
            except Exception as ffmpeg_error:
                raise RuntimeError(
                    f"Failed to read clip from {path} @ {start_frame}+{length}; "
                    f"pyav={pyav_error!r}; ffmpeg={ffmpeg_error!r}"
                ) from ffmpeg_error


class DroidInfidata(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {"1.1.0": "Semantic-preserving InfiData DROID RLDS with train/test splits and metadata."}

    def __init__(
        self,
        *,
        infidata_root: Path,
        lerobot_root: Path,
        train_episodes: list[InfiEpisode],
        unseen_test_episodes: list[InfiEpisode],
        seen_test_episodes: list[InfiEpisode],
        state_dim: int,
        action_dim: int,
        robots_json: Any,
        skip_log_path: Path,
        **kwargs: Any,
    ):
        self._infidata_root = Path(infidata_root)
        self._lerobot_root = Path(lerobot_root)
        self._train_episodes = train_episodes
        self._unseen_test_episodes = unseen_test_episodes
        self._seen_test_episodes = seen_test_episodes
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._robots_json = robots_json
        self._skip_log_path = Path(skip_log_path)
        self._original_index: dict[int, OriginalEpisode] | None = None
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        image = tfds.features.Image(shape=(180, 320, 3), dtype=np.uint8, encoding_format="jpeg")
        return tfds.core.DatasetInfo(
            builder=self,
            description="InfiData DROID converted to semantic-preserving RLDS.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "episode_index": np.int64,
                            "source_episode_index": np.int64,
                            "frame_index": np.int64,
                            "timestamp": np.float32,
                            "observation": {
                                "state": tfds.features.Tensor(shape=(self._state_dim,), dtype=np.float32),
                                "images": {camera: image for camera in CAMERAS},
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
                            "domain": tfds.features.Text(),
                            "subtask_annotation_status": tfds.features.Text(),
                            "task_category": tfds.features.Text(),
                            "building": tfds.features.Text(),
                            "collector_id": tfds.features.Text(),
                            "date": tfds.features.Text(),
                            "video": {
                                camera: {
                                    "path": tfds.features.Text(),
                                    "frame_index": np.int64,
                                }
                                for camera in CAMERAS
                            },
                            "subgoal": {
                                "real_future": {
                                    camera: {
                                        "path": tfds.features.Text(),
                                        "frame_index": np.int64,
                                    }
                                    for camera in CAMERAS
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
                        "source_episode_index": np.int64,
                        "source_file_index": np.int64,
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
                        "domain": tfds.features.Text(),
                        "video_paths_json": tfds.features.Text(),
                        "camera_mapping_json": tfds.features.Text(),
                        "state_action_schema_json": tfds.features.Text(),
                        "state_representation": tfds.features.Text(),
                        "action_representation": tfds.features.Text(),
                        "action_is_delta": np.bool_,
                        "raw_episode_metadata_json": tfds.features.Text(),
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

    def _get_original_index(self) -> dict[int, OriginalEpisode]:
        if self._original_index is None:
            self._original_index = _load_original_episode_index(self._lerobot_root)
        return self._original_index

    def _video_path(self, infi_camera: str, episode: OriginalEpisode) -> Path:
        source_camera = SOURCE_CAMERA_BY_INFI[infi_camera]
        return (
            self._lerobot_root
            / "videos"
            / source_camera
            / f"chunk-{episode.video_chunk_indices[infi_camera]:03d}"
            / f"file-{episode.video_file_indices[infi_camera]:03d}.mp4"
        )

    def _metadata_value(self, episode: InfiEpisode, key: str, fallback: Any) -> Any:
        return episode.raw_json.get(key, fallback)

    def _episode_metadata(self, episode: InfiEpisode, rows: pd.DataFrame) -> dict[str, Any]:
        first = rows.iloc[0]
        return {
            "episode_index": episode.episode_index,
            "source_episode_index": episode.source_episode_index,
            "source_file_index": int(self._metadata_value(episode, "source_file_index", -1)),
            "parquet_path": episode.parquet_path,
            "num_frames": int(self._metadata_value(episode, "num_frames", len(rows))),
            "fps": int(self._metadata_value(episode, "fps", 0)),
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
            "domain": _as_text(first.get("domain"), _as_text(self._metadata_value(episode, "domain", ""))),
            "video_paths_json": _json_dumps(self._metadata_value(episode, "video_paths", {})),
            "camera_mapping_json": _json_dumps(self._metadata_value(episode, "camera_mapping", {})),
            "state_action_schema_json": _json_dumps(DROID_STATE_ACTION_SCHEMA),
            "state_representation": DROID_STATE_ACTION_SCHEMA["state_representation"],
            "action_representation": DROID_STATE_ACTION_SCHEMA["action_representation"],
            "action_is_delta": DROID_STATE_ACTION_SCHEMA["action_is_delta"],
            "raw_episode_metadata_json": _json_dumps(episode.raw_json),
            "robots_json": _json_dumps(self._robots_json),
            "tasks_json_path": str(self._infidata_root / "meta" / "tasks.json"),
            "stats_json_path": str(self._infidata_root / "meta" / "stats.json"),
        }

    def _generate_examples(self, split_name: str, episodes: list[InfiEpisode]):
        original_index = self._get_original_index()
        total_frames = sum(episode.num_frames for episode in episodes)
        started = time.time()
        completed_frames = 0
        progress = tqdm.tqdm(
            total=len(episodes),
            desc=f"Converting DROID {split_name} ({total_frames:,} frames)",
            unit="ep",
            dynamic_ncols=True,
        )
        video_cache = VideoReaderCache()

        try:
            for episode in episodes:
                original = original_index.get(episode.source_episode_index)
                if original is None:
                    progress.update(1)
                    continue

                rows = pd.read_parquet(self._infidata_root / episode.parquet_path).reset_index(drop=True)
                if rows.empty:
                    progress.update(1)
                    continue
                length = len(rows)
                if length != original.length:
                    raise RuntimeError(
                        f"Episode {episode.episode_index} has {length} InfiData rows, "
                        f"but original metadata says {original.length}"
                    )

                clips = {}
                failed = False
                for camera in CAMERAS:
                    path = self._video_path(camera, original)
                    try:
                        clips[camera] = video_cache.read_clip(path, original.local_video_starts[camera], length)
                    except Exception as exc:
                        append_skip(
                            self._skip_log_path,
                            {
                                "dataset": "DROID",
                                "split": split_name,
                                "episode_index": episode.episode_index,
                                "source_episode_index": episode.source_episode_index,
                                "camera": camera,
                                "video_path": str(path),
                                "start_frame": original.local_video_starts[camera],
                                "length": length,
                                "error": repr(exc),
                            },
                        )
                        print(
                            f"[skip] DROID {split_name} episode={episode.episode_index} "
                            f"camera={camera}: {exc}",
                            flush=True,
                        )
                        failed = True
                        break
                if failed:
                    progress.update(1)
                    continue

                steps = []
                for i, row in rows.iterrows():
                    is_last = i == length - 1
                    step = {
                        "episode_index": int(row.get("episode_index", episode.episode_index)),
                        "source_episode_index": int(row.get("source_episode_index", episode.source_episode_index)),
                        "frame_index": int(row.get("frame_index", i)),
                        "timestamp": np.float32(row.get("timestamp", 0.0)),
                        "observation": {
                            "state": _as_float_array(row["observation.state"], (self._state_dim,)),
                            "images": {camera: clips[camera][i] for camera in CAMERAS},
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
                            for camera in CAMERAS
                        },
                        "subgoal": {
                            "real_future": {
                                camera: {
                                    "path": _as_text(row.get(f"subgoal.real_future.{camera}.path"), ""),
                                    "frame_index": int(row.get(f"subgoal.real_future.{camera}.frame_index", i)),
                                }
                                for camera in CAMERAS
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
                    eta=f"{remaining_frames / max(frames_per_s, 1e-6) / 60:.1f}",
                )
                progress.update(1)
        finally:
            video_cache.close()
            progress.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infidata-root", type=Path, default=Path("/mnt/workspace/InfiData/DROID"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("/mnt/workspace/wudi/ELUBrain/DROID"))
    parser.add_argument("--data-dir", type=Path, required=True, help="TFDS output parent directory.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument(
        "--validation-ratio",
        "--unseen-test-ratio",
        dest="unseen_test_ratio",
        type=float,
        default=0.05,
        help="Held-out unseen test ratio sampled from all episodes.",
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

    selected_episodes = _load_infidata_episodes(args.infidata_root, args.max_episodes, args.start_episode)
    if not selected_episodes:
        raise ValueError("No InfiData DROID episodes selected.")

    train_episodes, unseen_test_episodes, seen_test_episodes = _split_episodes(
        selected_episodes,
        unseen_test_ratio=args.unseen_test_ratio,
        seen_test_ratio=args.seen_test_ratio,
        seed=args.seed,
    )

    state_dim = _infer_vector_dim(args.infidata_root, selected_episodes[0], "observation.state")
    action_dim = _infer_vector_dim(args.infidata_root, selected_episodes[0], "action")
    total_frames = sum(episode.num_frames for episode in selected_episodes)
    print(
        f"Selected {len(selected_episodes):,} DROID episodes, {total_frames:,} frames, "
        f"state_dim={state_dim}, action_dim={action_dim}."
    )
    print(
        f"Split: train={len(train_episodes):,}, unseen_test={len(unseen_test_episodes):,}, "
        f"seen_test={len(seen_test_episodes):,}, unseen_test_ratio={args.unseen_test_ratio:.3f}, "
        f"seen_test_ratio={args.seen_test_ratio:.3f}, seed={args.seed}."
    )

    if args.overwrite:
        dataset_dir = args.data_dir / "droid_infidata"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        skip_log_path = args.data_dir / "skipped_episodes.jsonl"
        if skip_log_path.exists():
            skip_log_path.unlink()
    else:
        skip_log_path = args.data_dir / "skipped_episodes.jsonl"

    builder = DroidInfidata(
        infidata_root=args.infidata_root,
        lerobot_root=args.lerobot_root,
        train_episodes=train_episodes,
        unseen_test_episodes=unseen_test_episodes,
        seen_test_episodes=seen_test_episodes,
        state_dim=state_dim,
        action_dim=action_dim,
        robots_json=_enrich_robots_json(_load_json(args.infidata_root / "meta" / "robots.json")),
        skip_log_path=skip_log_path,
        data_dir=str(args.data_dir),
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            register_checksums=False,
            force_checksums_validation=False,
        ),
        download_dir=str(args.data_dir / "downloads"),
    )
    print(f"Built {builder.name}:{builder.version} at {builder.data_dir}")


if __name__ == "__main__":
    main()
