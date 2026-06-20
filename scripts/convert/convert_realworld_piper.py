"""Normalize realworld_piper LeRobot datasets into the InfiData intermediate layout.

The source directory contains multiple already-valid LeRobot v2.1 task datasets.
This script keeps the original videos by symlink, rewrites only the per-episode
parquet files with missing InfiData semantic columns, and writes global InfiData
metadata under one output root.

Example smoke test:
    python scripts/convert/convert_realworld_piper.py \
        --source_root /mnt/workspace/InfiData/realworld_piper \
        --out_root /mnt/workspace/tmp/realworld_piper_infidata_smoke \
        --num_episodes 2 \
        --overwrite

Full conversion:
    python scripts/convert/convert_realworld_piper.py \
        --source_root /mnt/workspace/InfiData/realworld_piper \
        --out_root /mnt/workspace/InfiData/realworld_piper_infidata \
        --overwrite
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

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


SOURCE_DATASET = "realworld_piper"
FUTURE_SECONDS = 2.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_index_from_path(path: Path) -> int:
    prefix = "episode_"
    stem = path.stem
    if not stem.startswith(prefix) or not stem[len(prefix) :].isdigit():
        raise ValueError(f"Unexpected episode parquet name: {path.name}")
    return int(stem[len(prefix) :])


def discover_dataset_roots(source_root: Path, patterns: list[str]) -> list[Path]:
    dataset_roots = sorted(path.parent.parent for path in source_root.glob("*/meta/info.json"))
    if patterns:
        dataset_roots = [
            path
            for path in dataset_roots
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
        ]
    if not dataset_roots:
        raise FileNotFoundError(f"No LeRobot datasets found under {source_root}")
    return dataset_roots


def infidata_camera_name(source_camera: str) -> str:
    lower = source_camera.lower()
    if "left_wrist" in lower:
        return "cam_left_wrist"
    if "right_wrist" in lower:
        return "cam_right_wrist"
    if "front" in lower:
        return "cam_front"
    if "head" in lower or "high" in lower:
        return "cam_high"
    suffix = source_camera.rsplit(".", 1)[-1]
    return f"cam_{suffix}"


def video_features(info: dict[str, Any]) -> list[str]:
    return [
        name
        for name, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def choose_camera_map(info: dict[str, Any]) -> dict[str, str]:
    mapping = {}
    for source_camera in video_features(info):
        mapping[infidata_camera_name(source_camera)] = source_camera
    return dict(sorted(mapping.items()))


def source_video_path(dataset_root: Path, info: dict[str, Any], source_camera: str, source_episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk_index = episode_chunk(source_episode_index, chunk_size=chunk_size)
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    rel = pattern.format(
        episode_chunk=chunk_index,
        video_key=source_camera,
        episode_index=source_episode_index,
    )
    return dataset_root / rel


def load_task_lookup(dataset_root: Path) -> dict[int, str]:
    lookup = {}
    for item in read_jsonl(dataset_root / "meta" / "tasks.jsonl"):
        task = str(item.get("task", "")).strip()
        if task:
            lookup[int(item["task_index"])] = task
    return lookup


def load_episode_lookup(dataset_root: Path) -> dict[int, dict[str, Any]]:
    return {
        int(item["episode_index"]): item
        for item in read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    }


def task_for_row(row: pd.Series, task_lookup: dict[int, str], fallback: str) -> str:
    task_index = int(row.get("task_index", 0))
    return task_lookup.get(task_index, fallback)


def episode_tasks(rows: pd.DataFrame, task_lookup: dict[int, str], fallback: str) -> list[str]:
    if "task_index" not in rows.columns:
        return [fallback]
    tasks = []
    for task_index in rows["task_index"].drop_duplicates().tolist():
        task = task_lookup.get(int(task_index), fallback)
        if task not in tasks:
            tasks.append(task)
    return tasks or [fallback]


def add_video_links(
    rows: pd.DataFrame,
    *,
    dataset_root: Path,
    dataset_name: str,
    info: dict[str, Any],
    camera_map: dict[str, str],
    output_episode_index: int,
    source_episode_index: int,
    out_root: Path,
    overwrite: bool,
    future_offset: int,
) -> dict[str, str]:
    video_paths = {}
    for infi_camera, source_camera in camera_map.items():
        src = source_video_path(dataset_root, info, source_camera, source_episode_index)
        rel = (
            Path("videos")
            / SOURCE_DATASET
            / dataset_name
            / infi_camera
            / f"episode_{output_episode_index:06d}.mp4"
        )
        safe_symlink_or_copy(src, out_root / rel, copy_video=False, overwrite=overwrite)
        rel_text = str(rel)
        video_paths[infi_camera] = rel_text
        rows[f"video.{infi_camera}.path"] = rel_text
        rows[f"video.{infi_camera}.frame_index"] = rows["frame_index"].astype("int64")
        rows[f"subgoal.real_future.{infi_camera}.path"] = rel_text
        rows[f"subgoal.real_future.{infi_camera}.frame_index"] = np.minimum(
            rows["frame_index"].to_numpy(dtype=np.int64) + future_offset,
            int(rows["frame_index"].iloc[-1]),
        ).astype(np.int64)
    return video_paths


def make_segments(rows: pd.DataFrame, output_episode_index: int, source_episode_index: int) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    segments = []
    current = str(rows["subtask"].iloc[0])
    start = 0
    for index in range(1, len(rows) + 1):
        boundary = index == len(rows) or str(rows["subtask"].iloc[index]) != current
        if not boundary:
            continue
        segments.append(
            {
                "episode_index": int(output_episode_index),
                "source_episode_index": int(source_episode_index),
                "segment_index": int(len(segments)),
                "start_frame": int(start),
                "end_frame": int(index - 1),
                "subtask": current,
                "task": str(rows["task"].iloc[start]),
                "annotation_status": "auto_placeholder",
                "source": "task_index",
            }
        )
        if index < len(rows):
            start = index
            current = str(rows["subtask"].iloc[index])
    return segments


def feature_names(feature: dict[str, Any] | None) -> list[str]:
    if not isinstance(feature, dict):
        return []
    names = feature.get("names")
    return [str(item) for item in names] if isinstance(names, list) else []


def schema_key(info: dict[str, Any], camera_map: dict[str, str], has_ee_pose: bool) -> str:
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    action_dim = int(info["features"]["action"]["shape"][0])
    suffix = "ee_pose" if has_ee_pose else "no_ee_pose"
    return f"piper_s{state_dim}_a{action_dim}_fps{int(info.get('fps', 0))}_c{len(camera_map)}_{suffix}"


def convert_episode(
    *,
    parquet_path: Path,
    dataset_root: Path,
    info: dict[str, Any],
    task_lookup: dict[int, str],
    episode_lookup: dict[int, dict[str, Any]],
    camera_map: dict[str, str],
    output_episode_index: int,
    out_root: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    source_episode_index = episode_index_from_path(parquet_path)
    rows = pd.read_parquet(parquet_path).sort_values(["frame_index", "timestamp"], kind="stable").reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"Empty episode parquet: {parquet_path}")

    expected = np.arange(len(rows))
    actual = rows["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError(f"Non-contiguous frame_index in {parquet_path}")

    fps = int(info.get("fps", 30))
    future_offset = max(1, int(round(FUTURE_SECONDS * fps)))
    fallback_task = dataset_root.name.replace("_", " ")
    tasks_by_row = [task_for_row(row, task_lookup, fallback_task) for _, row in rows.iterrows()]
    primary_task = tasks_by_row[0]
    source_episode_meta = episode_lookup.get(source_episode_index, {})

    rows["episode_index"] = int(output_episode_index)
    rows["source_episode_index"] = int(source_episode_index)
    rows["source_dataset_name"] = dataset_root.name
    rows["task"] = tasks_by_row
    rows["subtask"] = tasks_by_row
    rows["robot_type"] = str(info.get("robot_type", "piper"))
    rows["control_mode"] = "joint"
    rows["quality"] = 3
    rows["speed_bin"] = speed_bin_from_T(len(rows))
    rows["mistake"] = False
    rows["success"] = True
    rows["source_dataset"] = SOURCE_DATASET
    rows["domain"] = "real"
    rows["subtask_annotation_status"] = "auto_placeholder"
    rows["quality_annotation_status"] = "auto_placeholder"
    rows["outcome_annotation_status"] = "auto_placeholder"

    video_paths = add_video_links(
        rows,
        dataset_root=dataset_root,
        dataset_name=dataset_root.name,
        info=info,
        camera_map=camera_map,
        output_episode_index=output_episode_index,
        source_episode_index=source_episode_index,
        out_root=out_root,
        overwrite=overwrite,
        future_offset=future_offset,
    )

    out_parquet = (
        out_root
        / "data"
        / SOURCE_DATASET
        / f"chunk-{episode_chunk(output_episode_index):03d}"
        / f"episode_{output_episode_index:06d}.parquet"
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if out_parquet.exists() and not overwrite:
        raise FileExistsError(f"Output parquet already exists: {out_parquet}")
    rows.to_parquet(out_parquet, index=False)

    state = np.stack(rows["observation.state"].map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy())
    action = np.stack(rows["action"].map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy())
    stats = vector_stats(state, action)
    stats.update(
        {
            "episode_index": int(output_episode_index),
            "source_episode_index": int(source_episode_index),
            "source_dataset_name": dataset_root.name,
            "has_ee_pose": bool("observation.ee_pose" in rows.columns and "action.ee_pose" in rows.columns),
        }
    )

    duration_sec = float(source_episode_meta.get("real_duration_s", len(rows) / fps))
    episode_meta = {
        "episode_index": int(output_episode_index),
        "source_episode_index": int(source_episode_index),
        "source_file": str(parquet_path),
        "source_dataset_name": dataset_root.name,
        "parquet_path": str(out_parquet.relative_to(out_root)),
        "task": primary_task,
        "tasks": episode_tasks(rows, task_lookup, fallback_task),
        "subtask_annotation_status": "auto_placeholder",
        "robot_type": str(info.get("robot_type", "piper")),
        "control_mode": "joint",
        "num_frames": int(len(rows)),
        "fps": int(fps),
        "duration_sec": duration_sec,
        "speed_bin": int(speed_bin_from_T(len(rows))),
        "quality": 3,
        "success": True,
        "mistake": False,
        "quality_annotation_status": "auto_placeholder",
        "outcome_annotation_status": "auto_placeholder",
        "source_dataset": SOURCE_DATASET,
        "domain": "real",
        "video_paths": video_paths,
        "camera_mapping": camera_map,
        "has_ee_pose": bool("observation.ee_pose" in rows.columns and "action.ee_pose" in rows.columns),
        "source_episode_metadata": source_episode_meta,
        "annotation_note": (
            "subtask=task and quality/outcome fields are auto placeholders. "
            "Original LeRobot state/action/video/ee_pose values are preserved."
        ),
    }
    return episode_meta, make_segments(rows, output_episode_index, source_episode_index), stats


def write_readme(out_root: Path, num_episodes: int, num_datasets: int, num_frames: int) -> None:
    text = f"""# realworld_piper InfiData

Normalized from the existing realworld_piper LeRobot v2.1 task datasets.

- Source task datasets normalized: {num_datasets}
- Episodes normalized: {num_episodes}
- Frames normalized: {num_frames}
- Domain: real
- Robot type: piper
- Control mode: joint
- Videos: symlinked, not copied or re-encoded

This normalization is intentionally light-weight. It preserves the source
`observation.state`, `action`, `observation.ee_pose`, `action.ee_pose`, and
real timestamp columns when present, while adding the InfiData semantic columns
and global metadata required by downstream RLDS conversion.

Action semantics inferred from local samples:
`action[t]` matches `observation.state[t+1]` for checked episodes, so the action
is recorded in `meta/robots.json` as a next-step absolute joint target rather
than a delta.

`quality`, `success`, `mistake`, and `subtask` are auto placeholders because no
reviewed outcome/subtask labels were present in the source LeRobot datasets.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize realworld_piper LeRobot datasets to InfiData.")
    parser.add_argument("--source_root", type=Path, default=Path("/mnt/workspace/InfiData/realworld_piper"))
    parser.add_argument("--out_root", type=Path, default=Path("/mnt/workspace/InfiData/realworld_piper_infidata"))
    parser.add_argument("--dataset_glob", action="append", default=[], help="Optional task dataset glob; may repeat.")
    parser.add_argument("--num_episodes", type=int, default=0, help="0 converts all episodes.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    dataset_roots = discover_dataset_roots(source_root, args.dataset_glob)

    all_episode_meta = []
    all_segments = []
    all_stats = []
    all_tasks = {}
    robot_configs = {}
    skipped = []
    converted_datasets = set()
    stop = False

    for dataset_root in tqdm(dataset_roots, desc="realworld_piper task datasets"):
        info = load_json(dataset_root / "meta" / "info.json")
        task_lookup = load_task_lookup(dataset_root)
        episode_lookup = load_episode_lookup(dataset_root)
        camera_map = choose_camera_map(info)
        episode_files = sorted(
            dataset_root.glob("data/chunk-*/episode_*.parquet"),
            key=episode_index_from_path,
        )

        for task_index, task in task_lookup.items():
            all_tasks[f"{dataset_root.name}:{task_index}"] = {
                "task": task,
                "source_dataset": SOURCE_DATASET,
                "source_dataset_name": dataset_root.name,
                "source_task_index": int(task_index),
            }

        for parquet_path in episode_files:
            if args.num_episodes > 0 and len(all_episode_meta) >= args.num_episodes:
                stop = True
                break
            try:
                episode_meta, segments, stats = convert_episode(
                    parquet_path=parquet_path,
                    dataset_root=dataset_root,
                    info=info,
                    task_lookup=task_lookup,
                    episode_lookup=episode_lookup,
                    camera_map=camera_map,
                    output_episode_index=len(all_episode_meta),
                    out_root=out_root,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                if args.strict:
                    raise
                skipped.append(
                    {
                        "source_dataset_name": dataset_root.name,
                        "source_file": str(parquet_path),
                        "reason": str(exc),
                    }
                )
                continue

            all_episode_meta.append(episode_meta)
            all_segments.extend(segments)
            all_stats.append(stats)
            converted_datasets.add(dataset_root.name)

            has_ee_pose = bool(episode_meta["has_ee_pose"])
            key = schema_key(info, camera_map, has_ee_pose)
            features = info["features"]
            state_feature = features.get("observation.state", {})
            action_feature = features.get("action", {})
            ee_pose_feature = features.get("observation.ee_pose", {})
            config = robot_configs.setdefault(
                key,
                {
                    "robot_type": str(info.get("robot_type", "piper")),
                    "domain": "real",
                    "state_dim": int(state_feature.get("shape", [0])[0]),
                    "action_dim": int(action_feature.get("shape", [0])[0]),
                    "control_mode": "joint",
                    "fps": int(info.get("fps", 0)),
                    "cameras": sorted(camera_map),
                    "camera_mapping": camera_map,
                    "has_ee_pose": has_ee_pose,
                    "state_feature": state_feature,
                    "action_feature": action_feature,
                    "ee_pose_feature": ee_pose_feature,
                    "source_dataset_names": set(),
                    "state_action_schema": {
                        "control_mode": "joint",
                        "state_representation": "absolute_joint_position_with_gripper",
                        "action_representation": "next_step_absolute_joint_position_with_gripper",
                        "action_is_delta": False,
                        "state_layout": feature_names(state_feature),
                        "action_layout": feature_names(action_feature),
                        "ee_pose_layout": feature_names(ee_pose_feature),
                        "ee_pose_coordinate_frame": "piper_robot_base_or_source_control_frame",
                        "ee_pose_note": (
                            "Source feature names provide XYZ in meters and RX/RY/RZ in radians. "
                            "No camera extrinsics were found, so ee_pose is not marked as camera-frame pose."
                        ),
                    },
                },
            )
            config["source_dataset_names"].add(dataset_root.name)
        if stop:
            break

    if not all_episode_meta:
        raise RuntimeError("No valid realworld_piper episodes were normalized.")

    robots = {}
    for key, config in robot_configs.items():
        item = dict(config)
        item["source_dataset_names"] = sorted(item["source_dataset_names"])
        robots[key] = item

    dataset_stats = {
        "num_source_datasets": len(converted_datasets),
        "num_episodes": len(all_episode_meta),
        "num_frames": int(sum(item["num_frames"] for item in all_episode_meta)),
        "episodes": all_stats,
        "skipped_episodes": skipped,
    }

    write_jsonl(out_root / "meta" / "episodes.jsonl", all_episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", all_segments)
    write_json(out_root / "meta" / "tasks.json", all_tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", dataset_stats)
    write_readme(
        out_root,
        num_episodes=len(all_episode_meta),
        num_datasets=len(converted_datasets),
        num_frames=dataset_stats["num_frames"],
    )

    print(f"[DONE] Normalized {len(all_episode_meta)} episodes to {out_root}")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} episodes; see meta/stats.json")


if __name__ == "__main__":
    main()
