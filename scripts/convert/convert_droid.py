import argparse
import json
from collections import defaultdict
from pathlib import Path

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


CAMERA_MAP = {
    "cam_high": "observation.images.exterior_1_left",
    "cam_left_wrist": "observation.images.wrist_left",
    "cam_right_wrist": "observation.images.exterior_2_left",
}


def parse_file_index(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def load_episode_metadata(droid_root: Path) -> pd.DataFrame:
    files = sorted(droid_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No DROID episode metadata found under {droid_root / 'meta/episodes'}")

    columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for source_camera in CAMERA_MAP.values():
        columns.extend(
            [
                f"videos/{source_camera}/chunk_index",
                f"videos/{source_camera}/file_index",
                f"videos/{source_camera}/from_timestamp",
                f"videos/{source_camera}/to_timestamp",
            ]
        )

    frames = []
    for path in tqdm(files, desc="Reading DROID episode metadata"):
        frames.append(pd.read_parquet(path, columns=columns))

    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata.drop_duplicates("episode_index", keep="first")
    return metadata.sort_values("episode_index", kind="stable").reset_index(drop=True)


def load_task_lookup(droid_root: Path) -> dict[int, str]:
    path = droid_root / "meta" / "tasks.parquet"
    if not path.exists():
        return {}

    tasks_df = pd.read_parquet(path)
    lookup = {}
    if "task_index" in tasks_df.columns:
        for index_value, row in tasks_df.iterrows():
            task_index = int(row["task_index"])
            if isinstance(index_value, str) and index_value.strip():
                lookup[task_index] = index_value.strip()
            elif "task" in tasks_df.columns and isinstance(row.get("task"), str):
                lookup[task_index] = row["task"].strip()
    return lookup


def first_non_empty(values, default: str) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def infer_task(ep_df: pd.DataFrame, metadata_row: pd.Series, task_lookup: dict[int, str]) -> str:
    if "language_instruction" in ep_df.columns:
        task = first_non_empty(ep_df["language_instruction"], "")
        if task:
            return task

    metadata_tasks = metadata_row.get("tasks")
    if isinstance(metadata_tasks, np.ndarray):
        metadata_tasks = metadata_tasks.tolist()
    if isinstance(metadata_tasks, (list, tuple)):
        task = first_non_empty(metadata_tasks, "")
        if task:
            return task

    if "task_index" in ep_df.columns:
        task_index = int(ep_df["task_index"].iloc[0])
        if task_index in task_lookup:
            return task_lookup[task_index]
    return "droid manipulation task"


def pick_vector(df: pd.DataFrame, primary: str, fallback_fields: tuple[str, str]) -> np.ndarray:
    if primary in df.columns:
        values = [np.asarray(value, dtype=np.float32).reshape(-1) for value in df[primary]]
        return np.stack(values)

    first, second = fallback_fields
    if first in df.columns and second in df.columns:
        values = []
        for first_value, second_value in zip(df[first], df[second], strict=True):
            values.append(
                np.concatenate(
                    [
                        np.asarray(first_value, dtype=np.float32).reshape(-1),
                        np.asarray(second_value, dtype=np.float32).reshape(-1),
                    ]
                )
            )
        return np.stack(values)
    raise KeyError(f"Cannot construct {primary}; missing {first} or {second}")


def source_data_path(droid_root: Path, chunk_index: int, file_index: int) -> Path:
    return droid_root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"


def source_video_path(
    droid_root: Path,
    source_camera: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    return (
        droid_root
        / "videos"
        / source_camera
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def link_episode_videos(
    metadata_row: pd.Series,
    droid_root: Path,
    out_root: Path,
    copy_video: bool,
    overwrite: bool,
) -> tuple[dict, dict]:
    relative_paths = {}
    start_frames = {}

    for infi_camera, source_camera in CAMERA_MAP.items():
        prefix = f"videos/{source_camera}"
        chunk_index = int(metadata_row[f"{prefix}/chunk_index"])
        file_index = int(metadata_row[f"{prefix}/file_index"])
        from_timestamp = float(metadata_row[f"{prefix}/from_timestamp"])

        src = source_video_path(droid_root, source_camera, chunk_index, file_index)
        rel = (
            Path("videos")
            / "droid"
            / infi_camera
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        safe_symlink_or_copy(
            src,
            out_root / rel,
            copy_video=copy_video,
            overwrite=overwrite,
        )
        relative_paths[infi_camera] = str(rel)
        start_frames[infi_camera] = from_timestamp

    return relative_paths, start_frames


def read_source_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = ["episode_index", "frame_index", "timestamp"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required DROID columns in {path}: {missing}")
    return df


def select_episode_rows(source_df: pd.DataFrame, metadata_row: pd.Series) -> pd.DataFrame:
    source_episode_index = int(metadata_row["episode_index"])
    ep_df = source_df[source_df["episode_index"] == source_episode_index].copy()
    if ep_df.empty and "index" in source_df.columns:
        start = int(metadata_row["dataset_from_index"])
        end = int(metadata_row["dataset_to_index"])
        ep_df = source_df[(source_df["index"] >= start) & (source_df["index"] < end)].copy()

    if ep_df.empty:
        raise ValueError(f"No rows found for DROID episode {source_episode_index}")

    ep_df = ep_df.sort_values(["frame_index", "timestamp"], kind="stable").reset_index(drop=True)
    expected_length = int(metadata_row.get("length", len(ep_df)))
    if len(ep_df) != expected_length:
        raise ValueError(
            f"DROID episode {source_episode_index} has {len(ep_df)} rows, "
            f"metadata says {expected_length}"
        )
    return ep_df


def convert_episode(
    ep_df: pd.DataFrame,
    metadata_row: pd.Series,
    output_episode_index: int,
    droid_root: Path,
    out_root: Path,
    task_lookup: dict[int, str],
    fps: int,
    copy_video: bool,
    no_videos: bool,
    overwrite: bool,
    future_seconds: float,
):
    source_episode_index = int(metadata_row["episode_index"])
    state = pick_vector(
        ep_df,
        "observation.state",
        ("observation.state.joint_position", "observation.state.gripper_position"),
    )
    action = pick_vector(
        ep_df,
        "action",
        ("action.joint_position", "action.gripper_position"),
    )
    if state.shape[0] != action.shape[0]:
        raise ValueError(f"State/action length mismatch for DROID episode {source_episode_index}")

    T = len(ep_df)
    task = infer_task(ep_df, metadata_row, task_lookup)
    success = True
    if "is_episode_successful" in ep_df.columns:
        success = bool(ep_df["is_episode_successful"].astype(bool).mode().iloc[0])

    quality = 5 if success else 2
    mistake = not success
    speed_bin = speed_bin_from_T(T)
    future_offset = max(1, int(round(future_seconds * fps)))

    video_paths = {}
    video_start_timestamps = {}
    if not no_videos:
        video_paths, video_start_timestamps = link_episode_videos(
            metadata_row=metadata_row,
            droid_root=droid_root,
            out_root=out_root,
            copy_video=copy_video,
            overwrite=overwrite,
        )

    rows = []
    for t, source_row in ep_df.iterrows():
        row = {
            "episode_index": int(output_episode_index),
            "source_episode_index": int(source_episode_index),
            "frame_index": int(t),
            "timestamp": float(t / fps),
            "observation.state": state[t].tolist(),
            "action": action[t].tolist(),
            "task": task,
            "subtask": task,
            "robot_type": "franka",
            "control_mode": "joint",
            "quality": int(quality),
            "speed_bin": int(speed_bin),
            "mistake": bool(mistake),
            "success": bool(success),
            "source_dataset": "DROID",
            "domain": "real",
            "subtask_annotation_status": "auto_placeholder",
        }

        for name in (
            "language_instruction_2",
            "language_instruction_3",
            "task_category",
            "building",
            "collector_id",
            "date",
        ):
            value = source_row.get(name)
            if isinstance(value, str) and value:
                row[name] = value

        for infi_camera, rel_path in video_paths.items():
            source_frame = int(round(video_start_timestamps[infi_camera] * fps)) + t
            row[f"video.{infi_camera}.path"] = rel_path
            row[f"video.{infi_camera}.frame_index"] = int(source_frame)
            row[f"subgoal.real_future.{infi_camera}.path"] = rel_path
            row[f"subgoal.real_future.{infi_camera}.frame_index"] = int(
                source_frame + min(future_offset, T - 1 - t)
            )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    chunk_index = episode_chunk(output_episode_index)
    parquet_path = (
        out_root
        / "data"
        / "droid"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{output_episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists() and not overwrite:
        raise FileExistsError(f"Output parquet already exists: {parquet_path}")
    out_df.to_parquet(parquet_path, index=False)

    episode_meta = {
        "episode_index": int(output_episode_index),
        "source_episode_index": int(source_episode_index),
        "source_file_index": int(metadata_row["data/file_index"]),
        "parquet_path": str(parquet_path.relative_to(out_root)),
        "task": task,
        "robot_type": "franka",
        "control_mode": "joint",
        "num_frames": int(T),
        "fps": int(fps),
        "duration_sec": float(T / fps),
        "speed_bin": int(speed_bin),
        "quality": int(quality),
        "success": bool(success),
        "mistake": bool(mistake),
        "source_dataset": "DROID",
        "domain": "real",
        "video_paths": video_paths,
        "camera_mapping": CAMERA_MAP if not no_videos else {},
        "annotation_note": "subtask=task is an auto_placeholder; outcome comes from is_episode_successful",
    }
    segments = [
        {
            "episode_index": int(output_episode_index),
            "segment_index": 0,
            "start_frame": 0,
            "end_frame": int(T - 1),
            "task": task,
            "subtask": task,
            "mistake": bool(mistake),
            "annotation_status": "auto_placeholder",
            "source_episode_index": int(source_episode_index),
        }
    ]
    stats = vector_stats(state, action)
    stats.update(
        {
            "episode_index": int(output_episode_index),
            "source_episode_index": int(source_episode_index),
        }
    )
    return episode_meta, segments, stats


def write_readme(out_root: Path, num_episodes: int, fps: int, no_videos: bool) -> None:
    video_note = "Video columns were omitted." if no_videos else "Packed source videos are linked or copied once and referenced with absolute frame offsets."
    text = f"""# DROID InfiData

Converted from DROID LeRobot v3 packed files.

- Episodes: {num_episodes}
- Robot: Franka
- Domain: real
- FPS: {fps}
- State/action: source `observation.state` and `action`
- Subtask: task-level auto placeholder
- {video_note}

Episode boundaries and video offsets are read from `meta/episodes`. Missing or
incomplete source files are recorded in `meta/stats.json` unless `--strict` is used.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert DROID LeRobot v3 data to InfiData.")
    parser.add_argument(
        "--droid_root",
        default="/mnt/workspace/szeluresearch/ELUBrain/DROID",
    )
    parser.add_argument("--out_root", default="/mnt/workspace/InfiData/DROID")
    parser.add_argument("--num_episodes", type=int, default=0, help="0 converts all available episodes")
    parser.add_argument("--start_episode", type=int, default=0, help="source episode_index lower bound")
    parser.add_argument("--future_seconds", type=float, default=2.0)
    parser.add_argument("--copy_video", action="store_true")
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    droid_root = Path(args.droid_root).resolve()
    out_root = Path(args.out_root).resolve()
    info_path = droid_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing DROID metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = int(info.get("fps", 15))

    metadata = load_episode_metadata(droid_root)
    metadata = metadata[metadata["episode_index"] >= args.start_episode]
    if args.num_episodes > 0:
        metadata = metadata.iloc[: args.num_episodes]
    task_lookup = load_task_lookup(droid_root)

    grouped = defaultdict(list)
    for _, row in metadata.iterrows():
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        grouped[key].append(row)

    all_episode_meta = []
    all_segments = []
    all_stats = []
    skipped = []

    for (chunk_index, file_index), episode_rows in tqdm(
        grouped.items(),
        desc="Converting DROID data files",
    ):
        data_path = source_data_path(droid_root, chunk_index, file_index)
        try:
            source_df = read_source_file(data_path)
        except Exception as exc:
            if args.strict:
                raise
            for row in episode_rows:
                skipped.append(
                    {
                        "source_episode_index": int(row["episode_index"]),
                        "source_file": str(data_path),
                        "reason": str(exc),
                    }
                )
            continue

        for metadata_row in episode_rows:
            try:
                ep_df = select_episode_rows(source_df, metadata_row)
                episode_meta, segments, stats = convert_episode(
                    ep_df=ep_df,
                    metadata_row=metadata_row,
                    output_episode_index=len(all_episode_meta),
                    droid_root=droid_root,
                    out_root=out_root,
                    task_lookup=task_lookup,
                    fps=fps,
                    copy_video=args.copy_video,
                    no_videos=args.no_videos,
                    overwrite=args.overwrite,
                    future_seconds=args.future_seconds,
                )
            except Exception as exc:
                if args.strict:
                    raise
                skipped.append(
                    {
                        "source_episode_index": int(metadata_row["episode_index"]),
                        "source_file": str(data_path),
                        "reason": str(exc),
                    }
                )
                continue

            all_episode_meta.append(episode_meta)
            all_segments.extend(segments)
            all_stats.append(stats)

    if not all_episode_meta:
        raise RuntimeError("No valid DROID episodes were converted")

    tasks = {
        str(index): {"task": task, "source_dataset": "DROID"}
        for index, task in enumerate(sorted({item["task"] for item in all_episode_meta}))
    }
    robots = {
        "franka": {
            "robot_type": "franka",
            "domain": "real",
            "state_dim": int(len(all_stats[0]["state_mean"])),
            "action_dim": int(len(all_stats[0]["action_mean"])),
            "control_mode": "joint",
            "fps": int(fps),
            "cameras": [] if args.no_videos else list(CAMERA_MAP),
        }
    }
    dataset_stats = {
        "num_episodes": len(all_episode_meta),
        "num_frames": int(sum(item["num_frames"] for item in all_episode_meta)),
        "episodes": all_stats,
        "skipped_episodes": skipped,
    }

    write_jsonl(out_root / "meta" / "episodes.jsonl", all_episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", all_segments)
    write_json(out_root / "meta" / "tasks.json", tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", dataset_stats)
    write_readme(out_root, len(all_episode_meta), fps=fps, no_videos=args.no_videos)

    print(f"[DONE] Converted {len(all_episode_meta)} DROID episodes to {out_root}")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} episodes; see meta/stats.json")


if __name__ == "__main__":
    main()
