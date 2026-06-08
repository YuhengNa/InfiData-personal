import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_TASK = "articulation_tasks/franka/close_the_electriccooker"
CAMERA_MAP = {
    "cam_high": "images.rgb.head",
    # InterData-A1 has one hand camera. InfiData currently has no generic
    # wrist-camera key, so use cam_right_wrist as a compatibility alias.
    "cam_right_wrist": "images.rgb.hand",
}
PRESERVED_SOURCE_FIELDS = [
    "head_camera_intrinsics",
    "hand_camera_intrinsics",
    "head_camera_to_robot_extrinsics",
    "hand_camera_to_robot_extrinsics",
    "states.gripper.pose",
    "states.ee_to_armbase_pose",
    "states.ee_to_robot_pose",
    "states.tcp_to_armbase_pose",
    "states.tcp_to_robot_pose",
    "states.robot_to_env_pose",
    "actions.gripper.openness",
    "actions.gripper.pose",
    "actions.ee_to_armbase_pose",
    "actions.ee_to_robot_pose",
    "actions.tcp_to_armbase_pose",
    "actions.tcp_to_robot_pose",
    "master_actions.joint.position",
    "master_actions.gripper.position",
    "master_actions.gripper.openness",
    "master_actions.gripper.pose",
]


def speed_bin_from_T(T: int, bin_size: int = 500) -> int:
    return int(round(T / bin_size) * bin_size)


def is_dataset_root(path: Path) -> bool:
    return (
        (path / "data").is_dir()
        and (path / "meta" / "info.json").is_file()
        and (path / "videos").is_dir()
    )


def find_dataset_root(root: Path, task: str | None) -> Path:
    root = root.resolve()
    if is_dataset_root(root):
        return root

    if task:
        candidate = root / task
        if is_dataset_root(candidate):
            return candidate.resolve()

        nested = candidate / candidate.name
        if is_dataset_root(nested):
            return nested.resolve()

        raise FileNotFoundError(f"Cannot find InterData-A1 task dataset under {candidate}")

    roots = sorted(
        info.parent.parent.resolve()
        for info in root.rglob("meta/info.json")
        if is_dataset_root(info.parent.parent)
    )
    roots = list(dict.fromkeys(roots))
    if len(roots) == 1:
        return roots[0]
    if not roots:
        raise FileNotFoundError(f"No LeRobot dataset root found under {root}")
    raise ValueError("Multiple task datasets found; pass --task explicitly")


def episode_index_from_path(path: Path) -> int:
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix) or not stem[len(prefix) :].isdigit():
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(stem[len(prefix) :])


def load_tasks(task_root: Path) -> dict[int, str]:
    task_path = task_root / "meta" / "tasks.jsonl"
    if not task_path.exists():
        raise FileNotFoundError(f"Missing task metadata: {task_path}")

    tasks = {}
    with task_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            tasks[int(record["task_index"])] = str(record["task"]).strip()
    if not tasks:
        raise ValueError(f"No task records found in {task_path}")
    return tasks


def list_episode_files(task_root: Path) -> list[Path]:
    files = sorted(
        task_root.glob("data/chunk-*/episode_*.parquet"),
        key=episode_index_from_path,
    )
    if not files:
        raise FileNotFoundError(f"No v2.1 episode parquet files found under {task_root / 'data'}")
    return files


def safe_symlink_or_copy(src: Path, dst: Path, copy_video: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source video: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_video:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def source_video_path(task_root: Path, source_episode_index: int, source_camera: str) -> Path:
    chunk = source_episode_index // 1000
    return (
        task_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / source_camera
        / f"episode_{source_episode_index:06d}.mp4"
    )


def to_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stack_vector_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        raise KeyError(f"Missing required InterData-A1 field: {name}")
    values = [np.atleast_1d(np.asarray(value, dtype=np.float32)) for value in df[name]]
    return np.stack(values)


def derive_qvel(state: np.ndarray, fps: int) -> np.ndarray:
    qvel = np.zeros_like(state, dtype=np.float32)
    if len(state) > 1:
        qvel[0] = (state[1] - state[0]) * fps
        qvel[1:] = (state[1:] - state[:-1]) * fps
    return qvel


def read_episode(
    parquet_path: Path,
    task_lookup: dict[int, str],
    fps: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    source_df = pd.read_parquet(parquet_path)
    required = [
        "states.joint.position",
        "states.gripper.position",
        "actions.joint.position",
        "actions.gripper.position",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    ]
    missing = [name for name in required if name not in source_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {parquet_path}: {missing}")

    source_df = source_df.sort_values(["frame_index", "timestamp"], kind="stable").reset_index(drop=True)
    T = len(source_df)
    if T == 0:
        raise ValueError(f"Empty episode parquet: {parquet_path}")

    source_episode_indices = {int(x) for x in source_df["episode_index"]}
    if len(source_episode_indices) != 1:
        raise ValueError(f"Multiple episode_index values in {parquet_path}: {source_episode_indices}")

    expected_frames = np.arange(T)
    actual_frames = source_df["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_frames, expected_frames):
        raise ValueError(f"Non-contiguous frame_index values in {parquet_path}")

    task_indices = {int(x) for x in source_df["task_index"]}
    if len(task_indices) != 1:
        raise ValueError(f"Multiple task_index values in one episode: {task_indices}")
    source_task_index = next(iter(task_indices))
    if source_task_index not in task_lookup:
        raise KeyError(f"Unknown task_index={source_task_index} in {parquet_path}")

    joint_state = stack_vector_column(source_df, "states.joint.position")
    gripper_state = stack_vector_column(source_df, "states.gripper.position")
    joint_action = stack_vector_column(source_df, "actions.joint.position")
    gripper_action = stack_vector_column(source_df, "actions.gripper.position")

    state = np.concatenate([joint_state, gripper_state], axis=1)
    action = np.concatenate([joint_action, gripper_action], axis=1)
    if state.shape != action.shape:
        raise ValueError(f"State/action shape mismatch in {parquet_path}: {state.shape} vs {action.shape}")

    qvel = derive_qvel(state, fps=fps)
    action_relative = action - state
    return (
        source_df,
        state,
        action,
        qvel,
        action_relative,
        task_lookup[source_task_index],
        source_task_index,
    )


def convert_episode(
    parquet_path: Path,
    task_root: Path,
    out_root: Path,
    output_episode_index: int,
    task_lookup: dict[int, str],
    info: dict,
    fps: int,
    copy_video: bool,
    future_seconds: float,
):
    source_episode_index = episode_index_from_path(parquet_path)
    (
        source_df,
        state,
        action,
        qvel,
        action_relative,
        task,
        source_task_index,
    ) = read_episode(parquet_path, task_lookup=task_lookup, fps=fps)

    T = len(source_df)
    speed_bin = speed_bin_from_T(T)
    source_category = task_root.parents[2].name if len(task_root.parents) >= 3 else "unknown"
    source_task_name = task_root.name

    video_paths = {}
    for infi_camera, source_camera in CAMERA_MAP.items():
        src = source_video_path(task_root, source_episode_index, source_camera)
        dst_name = f"episode_{output_episode_index:06d}_{infi_camera}.mp4"
        dst = out_root / "videos" / "interdata_a1" / dst_name
        safe_symlink_or_copy(src, dst, copy_video=copy_video)
        video_paths[infi_camera] = str(Path("videos") / "interdata_a1" / dst_name)

    future_offset = max(1, int(round(future_seconds * fps)))
    rows = []
    for frame_idx, source_row in source_df.iterrows():
        row = {
            "episode_index": int(output_episode_index),
            "source_episode_index": int(source_episode_index),
            "frame_index": int(frame_idx),
            "timestamp": float(source_row["timestamp"]),
            "observation.state": state[frame_idx].tolist(),
            "observation.qvel": qvel[frame_idx].tolist(),
            "observation.qvel_source": "finite_difference_from_observation.state",
            "action": action[frame_idx].tolist(),
            "action_relative": action_relative[frame_idx].tolist(),
            "task": task,
            "subtask": task,
            "robot_type": "franka",
            "source_robot_type": str(info.get("robot_type", "unknown")),
            "control_mode": "joint",
            "quality": 3,
            "speed_bin": int(speed_bin),
            "mistake": False,
            "success": True,
            "quality_annotation_status": "auto_placeholder",
            "outcome_annotation_status": "auto_placeholder",
            "source_dataset": "InterData-A1",
            "source_release": "sim_updated",
            "source_category": source_category,
            "source_task_name": source_task_name,
            "source_task_index": int(source_task_index),
            "domain": "sim",
        }

        for name in PRESERVED_SOURCE_FIELDS:
            if name in source_df.columns:
                row[name] = to_native(source_row[name])

        for infi_camera, rel_path in video_paths.items():
            row[f"video.{infi_camera}.path"] = rel_path
            row[f"video.{infi_camera}.frame_index"] = int(source_row["frame_index"])
            row[f"subgoal.real_future.{infi_camera}.path"] = rel_path
            row[f"subgoal.real_future.{infi_camera}.frame_index"] = min(
                int(frame_idx + future_offset), T - 1
            )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    parquet_out = (
        out_root
        / "data"
        / "interdata_a1"
        / "chunk-000"
        / f"episode_{output_episode_index:06d}.parquet"
    )
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(parquet_out, index=False)

    episode_meta = {
        "episode_index": int(output_episode_index),
        "source_episode_index": int(source_episode_index),
        "source_file": str(parquet_path),
        "parquet_path": str(parquet_out.relative_to(out_root)),
        "task": task,
        "subtask_annotation_status": "auto_placeholder",
        "robot_type": "franka",
        "source_robot_type": str(info.get("robot_type", "unknown")),
        "control_mode": "joint",
        "num_frames": int(T),
        "fps": int(fps),
        "duration_sec": float(T / fps),
        "speed_bin": int(speed_bin),
        "quality": 3,
        "success": True,
        "mistake": False,
        "quality_annotation_status": "auto_placeholder",
        "outcome_annotation_status": "auto_placeholder",
        "source_dataset": "InterData-A1",
        "source_release": "sim_updated",
        "source_category": source_category,
        "source_task_name": source_task_name,
        "source_task_index": int(source_task_index),
        "domain": "sim",
        "video_paths": video_paths,
        "camera_mapping": CAMERA_MAP,
        "annotation_note": (
            "subtask=task and quality/outcome fields are unreviewed placeholders. "
            "Do not treat them as approved annotations."
        ),
    }

    segment_meta = [
        {
            "episode_index": int(output_episode_index),
            "segment_index": 0,
            "start_frame": 0,
            "end_frame": int(T - 1),
            "task": task,
            "subtask": task,
            "mistake": False,
            "annotation_status": "auto_placeholder",
            "source_episode_index": int(source_episode_index),
        }
    ]

    stats = {
        "episode_index": int(output_episode_index),
        "source_episode_index": int(source_episode_index),
        "num_frames": int(T),
        "state_min": state.min(axis=0).tolist(),
        "state_max": state.max(axis=0).tolist(),
        "state_mean": state.mean(axis=0).tolist(),
        "state_std": state.std(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": action.std(axis=0).tolist(),
    }
    return episode_meta, segment_meta, stats


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_readme(
    out_root: Path,
    task_root: Path,
    num_episodes: int,
    fps: int,
    future_seconds: float,
) -> None:
    text = f"""# interdata_a1_sim_updated_mini

This is a small InfiData-style dataset converted from InterData-A1 `sim_updated`
(LeRobot v2.1).

## Contents

- Converted task root: `{task_root}`
- Number of episodes: {num_episodes}
- Robot: Franka single arm
- Domain: simulation
- Control mode: joint
- FPS: {fps}
- State: 7 joint positions + 1 gripper position
- Action: 7 joint targets + 1 gripper target
- Future visual target offset: {future_seconds:g} seconds

## Camera mapping

- `images.rgb.head` -> `video.cam_high.*`
- `images.rgb.hand` -> `video.cam_right_wrist.*`

InterData-A1 has one generic hand camera. `cam_right_wrist` is only an InfiData
schema compatibility alias and does not assert physical right-arm placement.

## Derived and preserved fields

- `observation.qvel` is a finite difference of `observation.state`.
- `action_relative = action - observation.state`, matching the InternVLA-A1
  delta-action convention.
- Camera calibration, EE/TCP poses, and `master_actions.*` are preserved as
  source-specific extension columns.
- `subgoal.real_future.*` points to a future frame in the same source video.

## Annotation warning

InterData-A1 does not provide InfiData `subtask`, `quality`, `mistake`, or
`success` annotations. This prototype writes:

- `subtask = task`
- `quality = 3`
- `mistake = false`
- `success = true`

These values use `auto_placeholder` provenance fields and are present only to
satisfy the current required InfiData schema. They must be replaced or reviewed
before quality-conditioned or failure-recovery training.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def prepare_output(out_root: Path, overwrite: bool) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_root}; pass --overwrite")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interdata_root",
        required=True,
        help="InterData-A1 sim_updated root or one LeRobot v2.1 task dataset root",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help=f"Task path relative to sim_updated root (default: {DEFAULT_TASK})",
    )
    parser.add_argument("--out_root", default="examples/interdata_a1_sim_updated_mini")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--future_seconds", type=float, default=2.0)
    parser.add_argument("--copy_video", action="store_true", help="copy videos instead of symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive")
    if args.future_seconds <= 0:
        raise ValueError("--future_seconds must be positive")

    task_root = find_dataset_root(Path(args.interdata_root), args.task)
    out_root = Path(args.out_root).resolve()
    prepare_output(out_root, overwrite=args.overwrite)

    info = json.loads((task_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"This converter expects LeRobot v2.1 sim_updated data, got "
            f"{info.get('codebase_version')!r}"
        )
    fps = int(info.get("fps", 30))
    task_lookup = load_tasks(task_root)
    episode_files = list_episode_files(task_root)
    selected = episode_files[: args.num_episodes]

    print(f"[INFO] Task root: {task_root}")
    print(f"[INFO] Source episodes: {len(episode_files)}")
    print(f"[INFO] Converting: {len(selected)}")
    print(f"[INFO] Output: {out_root}")

    episode_meta = []
    segment_meta = []
    episode_stats = []
    for output_episode_index, parquet_path in enumerate(tqdm(selected, desc="Converting episodes")):
        ep, segments, stats = convert_episode(
            parquet_path=parquet_path,
            task_root=task_root,
            out_root=out_root,
            output_episode_index=output_episode_index,
            task_lookup=task_lookup,
            info=info,
            fps=fps,
            copy_video=args.copy_video,
            future_seconds=args.future_seconds,
        )
        episode_meta.append(ep)
        segment_meta.extend(segments)
        episode_stats.append(stats)

    used_tasks = sorted({record["task"] for record in episode_meta})
    tasks = {
        str(index): {
            "task": task,
            "source_dataset": "InterData-A1",
            "source_release": "sim_updated",
            "annotation_status": "source_metadata",
        }
        for index, task in enumerate(used_tasks)
    }
    robots = {
        "franka": {
            "robot_type": "franka",
            "source_robot_type": str(info.get("robot_type", "unknown")),
            "domain": "sim",
            "state_dim": 8,
            "action_dim": 8,
            "control_mode": "joint",
            "fps": fps,
            "cameras": list(CAMERA_MAP),
            "camera_mapping": CAMERA_MAP,
            "state_description": "7 joint positions + 1 gripper position",
            "action_description": "7 joint targets + 1 gripper target",
        }
    }
    stats = {
        "num_episodes": len(episode_meta),
        "num_frames": int(sum(record["num_frames"] for record in episode_meta)),
        "episodes": episode_stats,
    }

    write_jsonl(out_root / "meta" / "episodes.jsonl", episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", segment_meta)
    write_json(out_root / "meta" / "tasks.json", tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", stats)
    write_readme(
        out_root=out_root,
        task_root=task_root,
        num_episodes=len(episode_meta),
        fps=fps,
        future_seconds=args.future_seconds,
    )

    print("[DONE] InterData-A1 sim_updated mini dataset built")


if __name__ == "__main__":
    main()
