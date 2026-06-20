import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from convert_common import (
    episode_chunk,
    safe_symlink_or_copy,
    speed_bin_from_T,
    vector_stats,
    write_json,
    write_jsonl,
)


DEFAULT_AGIBOT_ROOT = Path(
    "/mnt/workspace/wudi/ELUBrain/AgiBotWorld-Beta-ModelScope-500h-extracted"
)
DEFAULT_METADATA_ROOT = Path(
    "/mnt/workspace/wudi/ELUBrain/AgiBotWorld-Beta-ModelScope-500h"
)
DEFAULT_OUT_ROOT = Path("/mnt/workspace/InfiData/AgiBotWorld-Beta-ModelScope-500h")
ROBOT_TYPE = "agibot_world_robot"
EMBODIMENT = "agibot_world_beta_mobile_dual_arm"
SOURCE_DATASET = "AgiBotWorld-Beta-ModelScope"
ACTION_TYPE = "absolute_position"
OUTCOME_ANNOTATION_STATUS = "auto_assumed_success_no_source_failure_label"

CAMERA_MAP = {
    "cam_high": "head_color.mp4",
    "cam_left_wrist": "hand_left_color.mp4",
    "cam_right_wrist": "hand_right_color.mp4",
}

STATE_FIELDS = [
    "state/joint/position",
    "state/effector/position",
    "state/head/position",
    "state/waist/position",
]
ACTION_FIELDS = [
    "action/joint/position",
    "action/effector/position",
    "action/head/position",
    "action/waist/position",
]


def shard_items(items: list, num_shards: int, shard_index: int) -> list:
    if num_shards <= 1:
        return items
    return [item for index, item in enumerate(items) if index % num_shards == shard_index]


def load_selected_task_ids(metadata_root: Path, selected_tasks_file: Path | None) -> list[int]:
    path = selected_tasks_file or metadata_root / "selection" / "selected_tasks.txt"
    if not path.exists():
        raise FileNotFoundError(f"selected tasks file not found: {path}")
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_task_info(metadata_root: Path, task_id: int) -> list[dict]:
    path = metadata_root / "task_info" / f"task_{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"task info not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"unexpected task info format: {path}")
    return records


def episode_paths(agibot_root: Path, task_id: int, episode_id: int) -> dict[str, Path]:
    return {
        "observation": agibot_root / "observations" / str(task_id) / str(episode_id),
        "parameters": agibot_root / "parameters" / str(task_id) / str(episode_id) / "parameters",
        "proprio": agibot_root
        / "proprio_stats"
        / str(task_id)
        / str(episode_id)
        / "proprio_stats.h5",
    }


def missing_episode_reasons(
    agibot_root: Path,
    task_id: int,
    episode_id: int,
    no_videos: bool,
) -> list[str]:
    paths = episode_paths(agibot_root, task_id, episode_id)
    reasons = []
    observation = paths["observation"]
    if not observation.is_dir():
        reasons.append("missing_observation_dir")
    elif not no_videos:
        video_dir = observation / "videos"
        for filename in CAMERA_MAP.values():
            video = video_dir / filename
            if not video.is_file() or video.stat().st_size == 0:
                reasons.append(f"missing_video:{filename}")
    if not paths["parameters"].is_dir():
        reasons.append("missing_parameters")
    if not paths["proprio"].is_file() or paths["proprio"].stat().st_size == 0:
        reasons.append("missing_proprio_stats")
    return reasons


def discover_episodes(
    agibot_root: Path,
    metadata_root: Path,
    selected_task_ids: list[int],
    no_videos: bool,
    drop_incomplete_tasks: bool,
    candidate_limit: int = 0,
) -> tuple[list[dict], list[dict], dict[int, dict]]:
    candidates = []
    skipped = []
    task_records = {}

    for task_id in selected_task_ids:
        records = load_task_info(metadata_root, task_id)
        task_records[task_id] = {int(record["episode_id"]): record for record in records}
        task_missing = []
        task_valid = []
        for record in records:
            episode_id = int(record["episode_id"])
            reasons = missing_episode_reasons(
                agibot_root=agibot_root,
                task_id=task_id,
                episode_id=episode_id,
                no_videos=no_videos,
            )
            item = {
                "task_id": task_id,
                "episode_id": episode_id,
                "task_name": str(record.get("task_name") or f"AgiBot task {task_id}"),
                "record": record,
            }
            if reasons:
                task_missing.append({**item, "reasons": reasons})
            else:
                task_valid.append(item)

        if drop_incomplete_tasks and task_missing:
            skipped.extend(
                {
                    "task_id": item["task_id"],
                    "episode_id": item["episode_id"],
                    "task_name": item["task_name"],
                    "reason": "drop_incomplete_tasks:" + ",".join(item["reasons"]),
                }
                for item in task_missing + task_valid
            )
            continue

        candidates.extend(task_valid)
        skipped.extend(
            {
                "task_id": item["task_id"],
                "episode_id": item["episode_id"],
                "task_name": item["task_name"],
                "reason": ",".join(item["reasons"]),
            }
            for item in task_missing
        )
        if candidate_limit > 0 and len(candidates) >= candidate_limit:
            candidates = candidates[:candidate_limit]
            break

    candidates.sort(key=lambda item: (item["task_id"], item["episode_id"]))
    return candidates, skipped, task_records


def read_concat(h5_file: h5py.File, names: list[str]) -> tuple[np.ndarray, list[str]]:
    missing = [name for name in names if name not in h5_file]
    if missing:
        raise KeyError(f"missing AgiBot datasets: {missing}")
    arrays = []
    dims = []
    for name in names:
        array = h5_file[name][:]
        if array.ndim != 2 or len(array) == 0:
            raise ValueError(f"unsupported or empty AgiBot dataset: {name} shape={array.shape}")
        arrays.append(array.astype(np.float32))
        dims.append(f"{name}:{array.shape[1]}")
    length = min(len(array) for array in arrays)
    return np.concatenate([array[:length] for array in arrays], axis=1), dims


def read_motion(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with h5py.File(h5_path, "r") as h5_file:
        state, state_dims = read_concat(h5_file, STATE_FIELDS)
        action, action_dims = read_concat(h5_file, ACTION_FIELDS)
        length = min(len(state), len(action))
        if length == 0:
            raise ValueError(f"empty AgiBot motion arrays: {h5_path}")
        state = state[:length]
        action = action[:length]

        if "timestamp" in h5_file and len(h5_file["timestamp"]) >= length:
            raw_timestamp = h5_file["timestamp"][:length].astype(np.float64)
            relative = raw_timestamp - raw_timestamp[0]
            if len(relative) > 1 and np.nanmedian(np.diff(raw_timestamp)) > 1_000_000:
                timestamp = relative / 1e9
            else:
                timestamp = relative
        else:
            timestamp = np.arange(length, dtype=np.float32) / 30.0

    meta = {
        "state_fields": STATE_FIELDS,
        "action_fields": ACTION_FIELDS,
        "state_dims": state_dims,
        "action_dims": action_dims,
    }
    return state, action, timestamp.astype(np.float32), meta


def make_segments(record: dict, task: str, episode_index: int, T: int) -> list[dict]:
    configs = record.get("label_info", {}).get("action_config", []) or []
    parsed = []
    for source_index, config in enumerate(configs):
        text = str(config.get("action_text") or "").strip()
        skill = str(config.get("skill") or "").strip()
        start = config.get("start_frame")
        end = config.get("end_frame")
        if start is None or end is None:
            continue
        start = max(0, min(int(start), T - 1))
        end = max(start, min(int(end), T - 1))
        parsed.append(
            {
                "start_frame": start,
                "end_frame": end,
                "subtask": text or task,
                "skill": skill,
                "source_action_config_index": int(source_index),
                "source_start_frame": int(config.get("start_frame")),
                "source_end_frame": int(config.get("end_frame")),
            }
        )
    parsed.sort(key=lambda item: (item["start_frame"], item["end_frame"]))

    if not parsed:
        return [
            {
                "episode_index": int(episode_index),
                "segment_index": 0,
                "start_frame": 0,
                "end_frame": int(T - 1),
                "task": task,
                "subtask": task,
                "skill": "",
                "annotation_status": "auto_placeholder",
            }
        ]

    segments = []
    cursor = 0
    for config in parsed:
        start = config["start_frame"]
        end = config["end_frame"]
        if start > cursor:
            segments.append(
                {
                    "episode_index": int(episode_index),
                    "segment_index": len(segments),
                    "start_frame": int(cursor),
                    "end_frame": int(start - 1),
                    "task": task,
                    "subtask": task,
                    "skill": "",
                    "annotation_status": "auto_gap_fill",
                }
            )
        start = max(start, cursor)
        if end >= start:
            segments.append(
                {
                    "episode_index": int(episode_index),
                    "segment_index": len(segments),
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "task": task,
                    "subtask": config["subtask"],
                    "skill": config["skill"],
                    "annotation_status": "source_agibot_action_config",
                    "source_action_config_index": config["source_action_config_index"],
                    "source_start_frame": config["source_start_frame"],
                    "source_end_frame": config["source_end_frame"],
                }
            )
            cursor = end + 1
    if cursor < T:
        segments.append(
            {
                "episode_index": int(episode_index),
                "segment_index": len(segments),
                "start_frame": int(cursor),
                "end_frame": int(T - 1),
                "task": task,
                "subtask": task,
                "skill": "",
                "annotation_status": "auto_gap_fill",
            }
        )
    return segments


def segment_for_frame(frame_index: int, segments: list[dict]) -> dict:
    for segment in segments:
        if segment["start_frame"] <= frame_index <= segment["end_frame"]:
            return segment
    return segments[-1]


def marker_path(out_root: Path, task_id: int, episode_id: int) -> Path:
    return out_root / ".conversion_markers" / f"task_{task_id}_episode_{episode_id}.json"


def load_marker_if_valid(out_root: Path, task_id: int, episode_id: int) -> tuple[dict, list[dict], dict] | None:
    path = marker_path(out_root, task_id, episode_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    parquet_path = out_root / payload["episode_meta"]["parquet_path"]
    if not parquet_path.exists():
        return None
    return payload["episode_meta"], payload["segments"], payload["stats"]


def write_marker(
    out_root: Path,
    task_id: int,
    episode_id: int,
    episode_meta: dict,
    segments: list[dict],
    stats: dict,
) -> None:
    path = marker_path(out_root, task_id, episode_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"episode_meta": episode_meta, "segments": segments, "stats": stats})


def link_videos(
    obs_dir: Path,
    out_root: Path,
    episode_index: int,
    copy_video: bool,
    overwrite: bool,
) -> dict[str, str]:
    video_paths = {}
    for infi_camera, source_filename in CAMERA_MAP.items():
        source = obs_dir / "videos" / source_filename
        rel = Path("videos") / "agibot" / f"episode_{episode_index:06d}_{infi_camera}.mp4"
        safe_symlink_or_copy(
            source,
            out_root / rel,
            copy_video=copy_video,
            overwrite=overwrite,
        )
        video_paths[infi_camera] = str(rel)
    return video_paths


def convert_episode(
    item: dict,
    episode_index: int,
    agibot_root: Path,
    out_root: Path,
    fps: int,
    future_seconds: float,
    no_videos: bool,
    copy_video: bool,
    overwrite: bool,
) -> tuple[dict, list[dict], dict]:
    task_id = int(item["task_id"])
    episode_id = int(item["episode_id"])
    existing = None if overwrite else load_marker_if_valid(out_root, task_id, episode_id)
    if existing is not None:
        return existing

    paths = episode_paths(agibot_root, task_id, episode_id)
    state, action, timestamp, motion_meta = read_motion(paths["proprio"])
    T = min(len(state), len(action), len(timestamp))
    state = state[:T]
    action = action[:T]
    timestamp = timestamp[:T]
    if T == 0:
        raise ValueError(f"empty AgiBot episode: task={task_id} episode={episode_id}")

    task = str(item["record"].get("task_name") or item["task_name"])
    init_scene_text = str(item["record"].get("init_scene_text") or "")
    segments = make_segments(item["record"], task=task, episode_index=episode_index, T=T)
    speed_bin = speed_bin_from_T(T)
    future_offset = max(1, int(round(future_seconds * fps)))
    video_paths = (
        {}
        if no_videos
        else link_videos(
            paths["observation"],
            out_root=out_root,
            episode_index=episode_index,
            copy_video=copy_video,
            overwrite=overwrite,
        )
    )

    rows = []
    for t in range(T):
        segment = segment_for_frame(t, segments)
        row = {
            "episode_index": int(episode_index),
            "frame_index": int(t),
            "timestamp": float(timestamp[t]),
            "observation.state": state[t].tolist(),
            "action": action[t].tolist(),
            "task": task,
            "subtask": segment["subtask"],
            "robot_type": ROBOT_TYPE,
            "embodiment": EMBODIMENT,
            "control_mode": "joint",
            "action_type": ACTION_TYPE,
            "quality": 5,
            "speed_bin": int(speed_bin),
            "mistake": False,
            "success": True,
            "quality_source": "default_success_demonstration_assumption",
            "success_source": "default_success_demonstration_assumption",
            "mistake_source": "default_success_demonstration_assumption",
            "speed_bin_source": "computed_from_num_frames_with_bin_size_500",
            "source_dataset": SOURCE_DATASET,
            "domain": "real",
            "source_task_id": int(task_id),
            "source_episode_id": int(episode_id),
            "source_task_name": task,
            "init_scene_text": init_scene_text,
            "subtask_skill": segment.get("skill", ""),
            "source_action_config_index": segment.get("source_action_config_index"),
            "subtask_annotation_status": segment["annotation_status"],
            "outcome_annotation_status": OUTCOME_ANNOTATION_STATUS,
        }
        for infi_camera, rel_path in video_paths.items():
            row[f"video.{infi_camera}.path"] = rel_path
            row[f"video.{infi_camera}.frame_index"] = int(t)
            row[f"subgoal.real_future.{infi_camera}.path"] = rel_path
            row[f"subgoal.real_future.{infi_camera}.frame_index"] = int(
                min(t + future_offset, T - 1)
            )
        rows.append(row)

    chunk_index = episode_chunk(episode_index)
    parquet_path = (
        out_root
        / "data"
        / "agibot"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output parquet exists without conversion marker: {parquet_path}"
        )
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    episode_meta = {
        "episode_index": int(episode_index),
        "source_task_id": int(task_id),
        "source_episode_id": int(episode_id),
        "source_proprio_file": str(paths["proprio"]),
        "source_observation_dir": str(paths["observation"]),
        "source_parameters_dir": str(paths["parameters"]),
        "parquet_path": str(parquet_path.relative_to(out_root)),
        "task": task,
        "source_task_name": task,
        "source_init_scene_text": init_scene_text,
        "source_label_info": item["record"].get("label_info", {}),
        "robot_type": ROBOT_TYPE,
        "embodiment": EMBODIMENT,
        "control_mode": "joint",
        "action_type": ACTION_TYPE,
        "num_frames": int(T),
        "fps": int(fps),
        "duration_sec": float(T / fps),
        "speed_bin": int(speed_bin),
        "quality": 5,
        "success": True,
        "mistake": False,
        "quality_source": "default_success_demonstration_assumption",
        "success_source": "default_success_demonstration_assumption",
        "mistake_source": "default_success_demonstration_assumption",
        "speed_bin_source": "computed_from_num_frames_with_bin_size_500",
        "source_dataset": SOURCE_DATASET,
        "domain": "real",
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "state_fields": motion_meta["state_fields"],
        "action_fields": motion_meta["action_fields"],
        "state_dims": motion_meta["state_dims"],
        "action_dims": motion_meta["action_dims"],
        "video_paths": video_paths,
        "camera_mapping": CAMERA_MAP if not no_videos else {},
        "annotation_note": (
            "State/action use AgiBot joint, effector, head, and waist position fields. "
            "Subtasks come from task_info label_info.action_config. "
            "AgiBot task_info in this subset does not expose per-episode failure labels; "
            "quality/success/mistake are schema defaults marked by *_source fields."
        ),
    }
    stats = vector_stats(state, action)
    stats["episode_index"] = int(episode_index)
    write_marker(out_root, task_id, episode_id, episode_meta, segments, stats)
    return episode_meta, segments, stats


def build_tasks_metadata(episode_meta: list[dict]) -> dict:
    tasks = {}
    for task_id in sorted({item["source_task_id"] for item in episode_meta}):
        task_texts = sorted(
            {item["task"] for item in episode_meta if item["source_task_id"] == task_id}
        )
        tasks[str(task_id)] = {
            "source_task_id": int(task_id),
            "task": task_texts[0] if task_texts else f"AgiBot task {task_id}",
            "source_task_name": task_texts[0] if task_texts else f"AgiBot task {task_id}",
            "source_dataset": SOURCE_DATASET,
        }
    return tasks


def build_robot_metadata(episode_meta: list[dict], fps: int, no_videos: bool) -> dict:
    return {
        ROBOT_TYPE: {
            "robot_type": ROBOT_TYPE,
            "domain": "real",
            "domains": sorted({item["domain"] for item in episode_meta}),
            "embodiment": EMBODIMENT,
            "embodiments": sorted({item["embodiment"] for item in episode_meta}),
            "state_dim": sorted({int(item["state_dim"]) for item in episode_meta})[0],
            "action_dim": sorted({int(item["action_dim"]) for item in episode_meta})[0],
            "state_dims": sorted({int(item["state_dim"]) for item in episode_meta}),
            "action_dims": sorted({int(item["action_dim"]) for item in episode_meta}),
            "control_mode": "joint",
            "action_type": ACTION_TYPE,
            "fps": int(fps),
            "cameras": [] if no_videos else list(CAMERA_MAP.keys()),
            "state_description": "AgiBot joint + effector + head + waist positions",
            "action_description": "AgiBot joint + effector + head + waist target positions",
            "source_state_fields": STATE_FIELDS,
            "source_action_fields": ACTION_FIELDS,
            "note": (
                "robot_type is a dataset-level generic AgiBot World robot label because "
                "task_info does not provide a per-episode hardware model identifier."
            ),
        }
    }


def write_readme(
    out_root: Path,
    num_episodes: int,
    selected_tasks: int,
    complete_tasks: int,
    fps: int,
    no_videos: bool,
    skipped: list[dict],
) -> None:
    text = f"""# AgiBotWorld-Beta ModelScope 500h Subset InfiData

Converted from the extracted AgiBotWorld-Beta ModelScope subset.

- Converted episodes: {num_episodes}
- Selected source tasks: {selected_tasks}
- Tasks with at least one converted episode: {complete_tasks}
- FPS: {fps}
- Videos linked: {not no_videos}
- Skipped/incomplete episodes: {len(skipped)}

The converter maps:

- `head_color.mp4` -> `cam_high`
- `hand_left_color.mp4` -> `cam_left_wrist`
- `hand_right_color.mp4` -> `cam_right_wrist`

`observation.state` and `action` are built from AgiBot joint, effector, head,
and waist position fields. Subtask segments are sourced from
`task_info/task_*.json` `label_info.action_config` entries.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert extracted AgiBotWorld-Beta ModelScope data to InfiData."
    )
    parser.add_argument("--agibot_root", type=Path, default=DEFAULT_AGIBOT_ROOT)
    parser.add_argument("--metadata_root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--selected_tasks_file", type=Path, default=None)
    parser.add_argument("--num_episodes", type=int, default=0, help="0 converts all valid episodes")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--episode_index_offset", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--future_seconds", type=float, default=2.0)
    parser.add_argument("--copy_video", action="store_true", help="copy videos instead of symlinking")
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument(
        "--drop_incomplete_tasks",
        action="store_true",
        help="drop all episodes from any task that has an incomplete episode",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    agibot_root = args.agibot_root.resolve()
    metadata_root = args.metadata_root.resolve()
    out_root = args.out_root.resolve()
    selected_task_ids = load_selected_task_ids(metadata_root, args.selected_tasks_file)
    candidate_limit = (
        args.num_episodes
        if args.num_episodes > 0
        and args.num_shards == 1
        and not args.drop_incomplete_tasks
        else 0
    )

    candidates, skipped, _task_records = discover_episodes(
        agibot_root=agibot_root,
        metadata_root=metadata_root,
        selected_task_ids=selected_task_ids,
        no_videos=args.no_videos,
        drop_incomplete_tasks=args.drop_incomplete_tasks,
        candidate_limit=candidate_limit,
    )
    candidates = shard_items(candidates, args.num_shards, args.shard_index)
    if args.num_episodes > 0:
        candidates = candidates[: args.num_episodes]

    if not candidates:
        raise RuntimeError("No valid AgiBot episodes found for conversion")

    all_episode_meta = []
    all_segments = []
    all_stats = []
    conversion_skipped = []

    for item in tqdm(candidates, desc="Converting AgiBot episodes"):
        episode_index = args.episode_index_offset + len(all_episode_meta)
        try:
            episode_meta, segments, stats = convert_episode(
                item=item,
                episode_index=episode_index,
                agibot_root=agibot_root,
                out_root=out_root,
                fps=args.fps,
                future_seconds=args.future_seconds,
                no_videos=args.no_videos,
                copy_video=args.copy_video,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            if args.strict:
                raise
            conversion_skipped.append(
                {
                    "task_id": int(item["task_id"]),
                    "episode_id": int(item["episode_id"]),
                    "task_name": item["task_name"],
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        all_episode_meta.append(episode_meta)
        all_segments.extend(segments)
        all_stats.append(stats)

    if not all_episode_meta:
        raise RuntimeError("No AgiBot episodes were converted")

    skipped_all = skipped + conversion_skipped
    tasks = build_tasks_metadata(all_episode_meta)
    robots = build_robot_metadata(all_episode_meta, fps=args.fps, no_videos=args.no_videos)
    dataset_stats = {
        "num_episodes": len(all_episode_meta),
        "num_frames": int(sum(item["num_frames"] for item in all_episode_meta)),
        "num_selected_tasks": int(len(selected_task_ids)),
        "num_tasks_with_converted_episodes": int(len(tasks)),
        "num_input_candidates": int(len(candidates)),
        "num_skipped_episodes": int(len(skipped_all)),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "episodes": all_stats,
        "skipped_episodes": skipped_all,
    }

    write_jsonl(out_root / "meta" / "episodes.jsonl", all_episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", all_segments)
    write_json(out_root / "meta" / "tasks.json", tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", dataset_stats)
    write_readme(
        out_root=out_root,
        num_episodes=len(all_episode_meta),
        selected_tasks=len(selected_task_ids),
        complete_tasks=len(tasks),
        fps=args.fps,
        no_videos=args.no_videos,
        skipped=skipped_all,
    )

    print(f"[DONE] Converted {len(all_episode_meta)} AgiBot episodes to {out_root}")
    if skipped_all:
        print(f"[INFO] Skipped {len(skipped_all)} episodes; see meta/stats.json")


if __name__ == "__main__":
    main()
