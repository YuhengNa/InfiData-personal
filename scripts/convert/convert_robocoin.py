import argparse
import fnmatch
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from convert_common import (
    episode_chunk,
    safe_symlink_or_copy,
    slugify,
    speed_bin_from_T,
    to_native,
    vector_stats,
    write_json,
    write_jsonl,
)


PRESERVED_SOURCE_FIELDS = [
    "scene_annotation",
    "eef_direction_state",
    "eef_velocity_state",
    "eef_acc_mag_state",
    "eef_direction_action",
    "eef_velocity_action",
    "eef_acc_mag_action",
    "eef_sim_pose_state",
    "eef_sim_pose_action",
    "gripper_mode_state",
    "gripper_mode_action",
    "gripper_activity_state",
    "gripper_activity_action",
]


def episode_index_from_path(path: Path) -> int:
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix) or not stem[len(prefix) :].isdigit():
        raise ValueError(f"Unexpected RoboCOIN episode filename: {path.name}")
    return int(stem[len(prefix) :])


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def discover_dataset_roots(root: Path, patterns: list[str]) -> list[Path]:
    dataset_roots = sorted(path.parent.parent for path in root.glob("*/meta/info.json"))
    if patterns:
        dataset_roots = [
            path
            for path in dataset_roots
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
        ]
    if not dataset_roots:
        raise FileNotFoundError(f"No RoboCOIN LeRobot v2.1 datasets found under {root}")
    return dataset_roots


def load_task_lookup(dataset_root: Path) -> dict[int, str]:
    lookup = {}
    for record in read_jsonl(dataset_root / "meta" / "tasks.jsonl"):
        task = record.get("task")
        if isinstance(task, str) and task.strip():
            lookup[int(record["task_index"])] = task.strip()
    return lookup


def load_episode_task_lookup(dataset_root: Path) -> dict[int, str]:
    lookup = {}
    for record in read_jsonl(dataset_root / "meta" / "episodes.jsonl"):
        tasks = record.get("tasks")
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, str) and task.strip():
                    lookup[int(record["episode_index"])] = task.strip()
                    break
    return lookup


def load_subtask_lookup(dataset_root: Path) -> dict[int, str]:
    lookup = {}
    path = dataset_root / "annotations" / "subtask_annotations.jsonl"
    for record in read_jsonl(path):
        subtask = record.get("subtask")
        if isinstance(subtask, str):
            lookup[int(record["subtask_index"])] = subtask.strip()
    return lookup


def infer_task(
    source_df: pd.DataFrame,
    source_episode_index: int,
    episode_tasks: dict[int, str],
    task_lookup: dict[int, str],
    dataset_name: str,
) -> str:
    if source_episode_index in episode_tasks:
        return episode_tasks[source_episode_index]
    if "task_index" in source_df.columns:
        task_index = int(source_df["task_index"].iloc[0])
        if task_index in task_lookup:
            return task_lookup[task_index]
    return dataset_name.replace("_", " ")


def video_features(info: dict) -> list[str]:
    return [
        name
        for name, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]


def first_matching(candidates: list[str], predicates) -> str | None:
    for predicate in predicates:
        for candidate in candidates:
            if predicate(candidate.lower()):
                return candidate
    return None


def choose_camera_map(info: dict) -> dict[str, str]:
    candidates = video_features(info)
    mapping = {}

    left = first_matching(
        candidates,
        [
            lambda name: "left" in name and "wrist" in name,
            lambda name: "color_left_wrist" in name,
        ],
    )
    right = first_matching(
        candidates,
        [
            lambda name: "right" in name and "wrist" in name,
            lambda name: "color_right_wrist" in name,
        ],
    )
    high = first_matching(
        candidates,
        [
            lambda name: name.endswith("cam_high_rgb"),
            lambda name: name.endswith("cam_head_rgb"),
            lambda name: name.endswith("cam_front_rgb"),
            lambda name: name.endswith("camera_head_rgb"),
            lambda name: name.endswith("ego_view"),
            lambda name: "high" in name and "wrist" not in name and "fisheye" not in name,
            lambda name: "head" in name and "wrist" not in name,
            lambda name: "front" in name and "wrist" not in name,
            lambda name: "wrist" not in name,
        ],
    )

    if high is not None:
        mapping["cam_high"] = high
    if left is not None:
        mapping["cam_left_wrist"] = left
    if right is not None:
        mapping["cam_right_wrist"] = right
    return mapping


def source_video_path(
    dataset_root: Path,
    info: dict,
    video_key: str,
    source_episode_index: int,
) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk_index = source_episode_index // chunk_size
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    relative = pattern.format(
        episode_chunk=episode_chunk_index,
        chunk_index=episode_chunk_index,
        episode_index=source_episode_index,
        video_key=video_key,
    )
    return dataset_root / relative


def link_episode_videos(
    dataset_root: Path,
    dataset_name: str,
    info: dict,
    camera_map: dict[str, str],
    source_episode_index: int,
    output_episode_index: int,
    out_root: Path,
    copy_video: bool,
    overwrite: bool,
) -> dict[str, str]:
    output = {}
    for infi_camera, source_camera in camera_map.items():
        src = source_video_path(dataset_root, info, source_camera, source_episode_index)
        rel = (
            Path("videos")
            / "robocoin"
            / slugify(dataset_name)
            / infi_camera
            / f"episode_{output_episode_index:06d}.mp4"
        )
        safe_symlink_or_copy(
            src,
            out_root / rel,
            copy_video=copy_video,
            overwrite=overwrite,
        )
        output[infi_camera] = str(rel)
    return output


def stack_vector_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        raise KeyError(f"Missing required RoboCOIN field: {name}")
    values = [np.asarray(value, dtype=np.float32).reshape(-1) for value in df[name]]
    return np.stack(values)


def decode_subtask_indices(value) -> list[int]:
    values = np.asarray(value).reshape(-1)
    return [int(item) for item in values]


def decode_subtask(
    value,
    subtask_lookup: dict[int, str],
    task: str,
) -> tuple[str, list[int], bool]:
    indices = decode_subtask_indices(value)
    labels = []
    used_indices = []
    abnormal = False

    for index in indices:
        label = subtask_lookup.get(index, "")
        normalized = label.strip().lower()
        if not normalized or normalized in {"null", "none", "nan"}:
            continue
        if normalized == "abnormal":
            abnormal = True
        if label not in labels:
            labels.append(label)
            used_indices.append(index)

    if not labels:
        return task, used_indices, abnormal
    return " | ".join(labels), used_indices, abnormal


def make_segments(
    frame_subtasks: list[str],
    frame_indices: list[list[int]],
    task: str,
    output_episode_index: int,
    source_episode_index: int,
    source_labeled: bool,
) -> list[dict]:
    segments = []
    start = 0
    current = frame_subtasks[0]
    current_indices = frame_indices[0]

    for frame_index in range(1, len(frame_subtasks) + 1):
        boundary = frame_index == len(frame_subtasks) or frame_subtasks[frame_index] != current
        if not boundary:
            continue
        segments.append(
            {
                "episode_index": int(output_episode_index),
                "segment_index": len(segments),
                "start_frame": int(start),
                "end_frame": int(frame_index - 1),
                "task": task,
                "subtask": current,
                "annotation_status": "human_labeled" if source_labeled else "auto_placeholder",
                "source_annotation_status": (
                    "robocoin_subtask_annotation" if source_labeled else "missing_source_subtask"
                ),
                "source_subtask_indices": current_indices,
                "source_episode_index": int(source_episode_index),
            }
        )
        if frame_index < len(frame_subtasks):
            start = frame_index
            current = frame_subtasks[frame_index]
            current_indices = frame_indices[frame_index]
    return segments


def source_failure(dataset_name: str, frame_subtasks: list[str]) -> bool:
    lowered = dataset_name.lower()
    if "fail" in lowered or "failure" in lowered:
        return True
    return any("abnormal" in subtask.lower() for subtask in frame_subtasks)


def convert_episode(
    parquet_path: Path,
    dataset_root: Path,
    info: dict,
    task_lookup: dict[int, str],
    episode_tasks: dict[int, str],
    subtask_lookup: dict[int, str],
    camera_map: dict[str, str],
    output_episode_index: int,
    out_root: Path,
    copy_video: bool,
    no_videos: bool,
    overwrite: bool,
    future_seconds: float,
):
    source_episode_index = episode_index_from_path(parquet_path)
    source_df = pd.read_parquet(parquet_path)
    if source_df.empty:
        raise ValueError(f"Empty RoboCOIN episode: {parquet_path}")

    source_df = source_df.sort_values(["frame_index", "timestamp"], kind="stable").reset_index(drop=True)
    if not np.array_equal(source_df["frame_index"].to_numpy(), np.arange(len(source_df))):
        raise ValueError(f"Non-contiguous frame_index in {parquet_path}")

    state = stack_vector_column(source_df, "observation.state")
    action = stack_vector_column(source_df, "action")
    T = len(source_df)
    dataset_name = dataset_root.name
    task = infer_task(
        source_df,
        source_episode_index,
        episode_tasks,
        task_lookup,
        dataset_name,
    )
    fps = int(info.get("fps", 30))
    speed_bin = speed_bin_from_T(T)
    future_offset = max(1, int(round(future_seconds * fps)))

    source_labeled = "subtask_annotation" in source_df.columns and bool(subtask_lookup)
    frame_subtasks = []
    frame_subtask_indices = []
    for _, row in source_df.iterrows():
        if source_labeled:
            subtask, indices, _ = decode_subtask(row["subtask_annotation"], subtask_lookup, task)
        else:
            subtask, indices = task, []
        frame_subtasks.append(subtask)
        frame_subtask_indices.append(indices)

    mistake = source_failure(dataset_name, frame_subtasks)
    success = not mistake
    quality = 2 if mistake else 3
    outcome_status = "source_inferred" if mistake else "auto_placeholder"

    video_paths = {}
    if not no_videos:
        video_paths = link_episode_videos(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            info=info,
            camera_map=camera_map,
            source_episode_index=source_episode_index,
            output_episode_index=output_episode_index,
            out_root=out_root,
            copy_video=copy_video,
            overwrite=overwrite,
        )

    rows = []
    for t, source_row in source_df.iterrows():
        row = {
            "episode_index": int(output_episode_index),
            "source_episode_index": int(source_episode_index),
            "frame_index": int(t),
            "timestamp": float(source_row.get("timestamp", t / fps)),
            "observation.state": state[t].tolist(),
            "action": action[t].tolist(),
            "task": task,
            "subtask": frame_subtasks[t],
            "robot_type": str(info.get("robot_type") or "unknown"),
            "control_mode": "joint",
            "quality": int(quality),
            "speed_bin": int(speed_bin),
            "mistake": bool(mistake),
            "success": bool(success),
            "source_dataset": "RoboCOIN",
            "source_dataset_name": dataset_name,
            "domain": "real",
            "subtask_annotation_status": (
                "human_labeled" if source_labeled else "auto_placeholder"
            ),
            "outcome_annotation_status": outcome_status,
        }

        if "subtask_annotation" in source_df.columns:
            row["source_subtask_annotation"] = decode_subtask_indices(
                source_row["subtask_annotation"]
            )
        for name in PRESERVED_SOURCE_FIELDS:
            if name in source_df.columns:
                row[name] = to_native(source_row[name])

        for infi_camera, rel_path in video_paths.items():
            row[f"video.{infi_camera}.path"] = rel_path
            row[f"video.{infi_camera}.frame_index"] = int(t)
            row[f"subgoal.real_future.{infi_camera}.path"] = rel_path
            row[f"subgoal.real_future.{infi_camera}.frame_index"] = int(
                min(t + future_offset, T - 1)
            )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    chunk_index = episode_chunk(output_episode_index)
    parquet_out = (
        out_root
        / "data"
        / "robocoin"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{output_episode_index:06d}.parquet"
    )
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    if parquet_out.exists() and not overwrite:
        raise FileExistsError(f"Output parquet already exists: {parquet_out}")
    out_df.to_parquet(parquet_out, index=False)

    segments = make_segments(
        frame_subtasks=frame_subtasks,
        frame_indices=frame_subtask_indices,
        task=task,
        output_episode_index=output_episode_index,
        source_episode_index=source_episode_index,
        source_labeled=source_labeled,
    )
    episode_meta = {
        "episode_index": int(output_episode_index),
        "source_episode_index": int(source_episode_index),
        "source_file": str(parquet_path),
        "source_dataset_name": dataset_name,
        "parquet_path": str(parquet_out.relative_to(out_root)),
        "task": task,
        "robot_type": str(info.get("robot_type") or "unknown"),
        "control_mode": "joint",
        "num_frames": int(T),
        "fps": int(fps),
        "duration_sec": float(T / fps),
        "speed_bin": int(speed_bin),
        "quality": int(quality),
        "success": bool(success),
        "mistake": bool(mistake),
        "source_dataset": "RoboCOIN",
        "domain": "real",
        "video_paths": video_paths,
        "camera_mapping": camera_map if not no_videos else {},
        "subtask_annotation_status": (
            "human_labeled" if source_labeled else "auto_placeholder"
        ),
        "outcome_annotation_status": outcome_status,
        "annotation_note": (
            "Subtasks are decoded from RoboCOIN subtask_annotation. "
            "Success/quality remain placeholders unless failure is explicit in source labels."
        ),
    }
    stats = vector_stats(state, action)
    stats.update(
        {
            "episode_index": int(output_episode_index),
            "source_episode_index": int(source_episode_index),
            "source_dataset_name": dataset_name,
        }
    )
    return episode_meta, segments, stats


def write_readme(out_root: Path, num_episodes: int, num_datasets: int, no_videos: bool) -> None:
    text = f"""# RoboCOIN InfiData

Converted from the RoboCOIN collection of LeRobot v2.1 task datasets.

- Source task datasets converted: {num_datasets}
- Episodes converted: {num_episodes}
- Domain: real
- Control mode: joint
- Videos included: {not no_videos}

The converter dynamically reads each task's robot type, state/action dimensions,
FPS, and camera features. RoboCOIN `subtask_annotation` values are decoded through
`annotations/subtask_annotations.jsonl` and merged into contiguous InfiData
segments. Source EEF, scene, and gripper annotations are retained when present.

`quality`, `success`, and `mistake` are not treated as reviewed labels. Explicit
failure task names or `Abnormal` subtask labels are marked as failures; other
episodes use successful-demo placeholders.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert RoboCOIN LeRobot v2.1 data to InfiData.")
    parser.add_argument(
        "--robocoin_root",
        default="/mnt/workspace/wudi/ELUBrain/RoboCOIN_data/RoboCOIN",
    )
    parser.add_argument("--out_root", default="/mnt/workspace/InfiData/RoboCOIN")
    parser.add_argument(
        "--dataset_glob",
        action="append",
        default=[],
        help="Optional task dataset glob; may be repeated",
    )
    parser.add_argument("--num_episodes", type=int, default=0, help="0 converts all episodes")
    parser.add_argument("--future_seconds", type=float, default=2.0)
    parser.add_argument("--copy_video", action="store_true")
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    robocoin_root = Path(args.robocoin_root).resolve()
    out_root = Path(args.out_root).resolve()
    dataset_roots = discover_dataset_roots(robocoin_root, args.dataset_glob)

    all_episode_meta = []
    all_segments = []
    all_stats = []
    skipped = []
    robot_configs = {}
    converted_datasets = set()

    stop = False
    for dataset_root in tqdm(dataset_roots, desc="RoboCOIN task datasets"):
        info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
        task_lookup = load_task_lookup(dataset_root)
        episode_tasks = load_episode_task_lookup(dataset_root)
        subtask_lookup = load_subtask_lookup(dataset_root)
        camera_map = choose_camera_map(info)
        episode_files = sorted(
            dataset_root.glob("data/chunk-*/episode_*.parquet"),
            key=episode_index_from_path,
        )

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
                    episode_tasks=episode_tasks,
                    subtask_lookup=subtask_lookup,
                    camera_map=camera_map,
                    output_episode_index=len(all_episode_meta),
                    out_root=out_root,
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

            config_key = slugify(
                f"{episode_meta['robot_type']}_s{len(stats['state_mean'])}"
                f"_a{len(stats['action_mean'])}_fps{episode_meta['fps']}"
            )
            config = robot_configs.setdefault(
                config_key,
                {
                    "robot_type": episode_meta["robot_type"],
                    "domain": "real",
                    "state_dim": int(len(stats["state_mean"])),
                    "action_dim": int(len(stats["action_mean"])),
                    "control_mode": "joint",
                    "fps": int(episode_meta["fps"]),
                    "cameras": set(),
                    "source_dataset_names": set(),
                },
            )
            config["cameras"].update(episode_meta["video_paths"])
            config["source_dataset_names"].add(dataset_root.name)
        if stop:
            break

    if not all_episode_meta:
        raise RuntimeError("No valid RoboCOIN episodes were converted")

    tasks = {
        str(index): {"task": task, "source_dataset": "RoboCOIN"}
        for index, task in enumerate(sorted({item["task"] for item in all_episode_meta}))
    }
    robots = {}
    for key, config in robot_configs.items():
        item = dict(config)
        item["cameras"] = sorted(item["cameras"])
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
    write_json(out_root / "meta" / "tasks.json", tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", dataset_stats)
    write_readme(
        out_root,
        num_episodes=len(all_episode_meta),
        num_datasets=len(converted_datasets),
        no_videos=args.no_videos,
    )

    print(f"[DONE] Converted {len(all_episode_meta)} RoboCOIN episodes to {out_root}")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} episodes; see meta/stats.json")


if __name__ == "__main__":
    main()
