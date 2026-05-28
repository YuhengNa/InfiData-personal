import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


CAMERA_MAP = {
    "cam_high": "observation.images.exterior_1_left",
    "cam_left_wrist": "observation.images.wrist_left",
    "cam_right_wrist": "observation.images.exterior_2_left",
}

MISSING_VIDEO_WARNED = set()


def speed_bin_from_T(T: int, bin_size: int = 500) -> int:
    return int(round(T / bin_size) * bin_size)


def parse_file_index_from_name(path: Path) -> int:
    # file-042.parquet -> 42
    stem = path.stem
    return int(stem.split("-")[-1])


def safe_symlink_or_copy(src: Path, dst: Path, copy_video: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return True

    if not src.exists():
        key = str(src)
        if key not in MISSING_VIDEO_WARNED:
            print(f"[WARN] video not found: {src}")
            MISSING_VIDEO_WARNED.add(key)
        return False

    if copy_video:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)
    return True


def make_placeholder_segments(T: int, task: str):
    cuts = [
        (0, int(T * 0.25), "approach the target object"),
        (int(T * 0.25), int(T * 0.50), "grasp or manipulate the object"),
        (int(T * 0.50), int(T * 0.75), "move toward the target location"),
        (int(T * 0.75), T, "finish the task"),
    ]

    segments = []
    for start, end, subtask in cuts:
        if end <= start:
            continue
        segments.append(
            {
                "start_frame": start,
                "end_frame": end - 1,
                "task": task,
                "subtask": subtask,
                "annotation_status": "auto_placeholder",
            }
        )
    return segments


def subtask_for_frame(frame_idx: int, segments) -> str:
    for seg in segments:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return seg["subtask"]
    return segments[-1]["subtask"] if segments else "unknown subtask"


def first_non_empty_text(series: pd.Series, default_text: str = "droid manipulation task") -> str:
    for x in series:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return default_text


def infer_task(df: pd.DataFrame) -> str:
    if "language_instruction" in df.columns:
        task = first_non_empty_text(df["language_instruction"])
        if task != "droid manipulation task":
            return task

    return "droid manipulation task"


def pick_state(df: pd.DataFrame) -> pd.Series:
    if "observation.state" in df.columns:
        return df["observation.state"]

    if "observation.state.joint_position" in df.columns and "observation.state.gripper_position" in df.columns:
        return df.apply(
            lambda r: list(np.asarray(r["observation.state.joint_position"], dtype=np.float32))
            + list(np.asarray(r["observation.state.gripper_position"], dtype=np.float32)),
            axis=1,
        )

    raise KeyError("Cannot find observation.state fields in DROID parquet.")


def pick_action(df: pd.DataFrame) -> pd.Series:
    if "action" in df.columns:
        return df["action"]

    if "action.joint_position" in df.columns and "action.gripper_position" in df.columns:
        return df.apply(
            lambda r: list(np.asarray(r["action.joint_position"], dtype=np.float32))
            + list(np.asarray(r["action.gripper_position"], dtype=np.float32)),
            axis=1,
        )

    if "action.original" in df.columns:
        return df["action.original"]

    raise KeyError("Cannot find action fields in DROID parquet.")


def iter_episode_groups(data_files):
    """Yield episode dataframes one by one to keep memory usage bounded."""
    for data_file in tqdm(data_files, desc="Scanning DROID parquet"):
        file_idx = parse_file_index_from_name(data_file)
        df = pd.read_parquet(data_file)

        required_cols = ["episode_index", "frame_index", "timestamp"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in {data_file}: {missing}")

        df = df.copy()
        df["_source_file_index"] = file_idx

        for ep_idx, g in df.groupby("episode_index", sort=False):
            yield int(ep_idx), g.reset_index(drop=True)


def convert_episode(
    ep_df: pd.DataFrame,
    source_episode_index: int,
    new_episode_index: int,
    droid_root: Path,
    out_root: Path,
    fps: int,
    copy_video: bool,
):
    ep_df = ep_df.sort_values(["timestamp", "frame_index"], kind="stable").reset_index(drop=True)

    state_series = pick_state(ep_df)
    action_series = pick_action(ep_df)

    T = len(ep_df)
    speed_bin = speed_bin_from_T(T)

    task = infer_task(ep_df)

    success = True
    if "is_episode_successful" in ep_df.columns:
        success = bool(ep_df["is_episode_successful"].astype(bool).mode().iloc[0])

    quality = 5 if success else 2
    mistake = not success

    segments = make_placeholder_segments(T, task)

    # Build reusable video path map keyed by source file index.
    file_indices = sorted(set(int(x) for x in ep_df["_source_file_index"].tolist()))
    linked_video = {}
    for file_idx in file_indices:
        linked_video[file_idx] = {}
        for infi_cam, droid_cam in CAMERA_MAP.items():
            src = droid_root / "videos" / droid_cam / "chunk-000" / f"file-{file_idx:03d}.mp4"
            dst_name = f"file-{file_idx:03d}_{infi_cam}.mp4"
            dst = out_root / "videos" / "droid" / dst_name
            ok = safe_symlink_or_copy(src, dst, copy_video=copy_video)
            linked_video[file_idx][infi_cam] = str(Path("videos") / "droid" / dst_name) if ok else None

    rows = []
    for local_idx, (_, r) in enumerate(ep_df.iterrows()):
        src_file_idx = int(r["_source_file_index"])
        row = {
            "episode_index": int(new_episode_index),
            "frame_index": int(local_idx),
            "timestamp": float(local_idx / fps),
            "observation.state": list(np.asarray(state_series.iloc[local_idx], dtype=np.float32)),
            "action": list(np.asarray(action_series.iloc[local_idx], dtype=np.float32)),
            "task": task,
            "subtask": subtask_for_frame(local_idx, segments),
            "robot_type": "franka",
            "control_mode": "joint",
            "quality": int(quality),
            "speed_bin": int(speed_bin),
            "mistake": bool(mistake),
            "success": bool(success),
            "source_dataset": "DROID",
            "domain": "real",
        }

        # Keep extra language fields when present for future annotation stages.
        if "language_instruction_2" in ep_df.columns and isinstance(r.get("language_instruction_2"), str):
            row["language_instruction_2"] = r.get("language_instruction_2")
        if "language_instruction_3" in ep_df.columns and isinstance(r.get("language_instruction_3"), str):
            row["language_instruction_3"] = r.get("language_instruction_3")

        for infi_cam in CAMERA_MAP.keys():
            row[f"video.{infi_cam}.path"] = linked_video[src_file_idx][infi_cam]
            row[f"video.{infi_cam}.frame_index"] = int(r.get("frame_index", local_idx))

        subgoal_t = min(local_idx + 2 * fps, T - 1)
        for infi_cam in CAMERA_MAP.keys():
            row[f"subgoal.real_future.{infi_cam}.path"] = linked_video[src_file_idx][infi_cam]
            row[f"subgoal.real_future.{infi_cam}.frame_index"] = int(subgoal_t)

        rows.append(row)

    out_df = pd.DataFrame(rows)

    parquet_path = out_root / "data" / "droid" / "chunk-000" / f"episode_{new_episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(parquet_path, index=False)

    episode_meta = {
        "episode_index": int(new_episode_index),
        "source_episode_index": int(source_episode_index),
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
        "video_keys": list(CAMERA_MAP.keys()),
        "annotation_note": "quality/mistake initialized from is_episode_successful; subtasks are placeholders",
    }

    segment_meta = []
    for i, seg in enumerate(segments):
        item = dict(seg)
        item["episode_index"] = int(new_episode_index)
        item["segment_index"] = int(i)
        segment_meta.append(item)

    state_arr = np.stack(out_df["observation.state"].to_numpy())
    action_arr = np.stack(out_df["action"].to_numpy())
    stats = {
        "episode_index": int(new_episode_index),
        "num_frames": int(T),
        "state_min": state_arr.min(axis=0).tolist(),
        "state_max": state_arr.max(axis=0).tolist(),
        "state_mean": state_arr.mean(axis=0).tolist(),
        "state_std": state_arr.std(axis=0).tolist(),
        "action_min": action_arr.min(axis=0).tolist(),
        "action_max": action_arr.max(axis=0).tolist(),
        "action_mean": action_arr.mean(axis=0).tolist(),
        "action_std": action_arr.std(axis=0).tolist(),
    }

    return episode_meta, segment_meta, stats


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_readme(out_root: Path, num_eps: int, fps: int):
    text = f"""# droid_mini

This is a small InfiData-style prototype dataset converted from DROID (LeRobot v3 format).

## Contents

- Number of converted episodes: {num_eps}
- Source dataset: DROID
- Robot: Franka
- Control mode: joint
- FPS: {fps}
- Cameras mapped to InfiData keys:
  - cam_high <- observation.images.exterior_1_left
  - cam_left_wrist <- observation.images.wrist_left
  - cam_right_wrist <- observation.images.exterior_2_left

## Important note

This is a prototype for data pipeline validation.

The fields `quality`, `success`, and `mistake` are initialized from `is_episode_successful`.
The `subtask` segments are auto-generated placeholders and should be replaced by human or VLM-assisted annotations before real training.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="examples/droid_mini")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--all_episodes", action="store_true", help="convert all episodes found in droid_root")
    parser.add_argument("--copy_video", action="store_true", help="copy videos instead of symlink")
    args = parser.parse_args()

    droid_root = Path(args.droid_root).resolve()
    out_root = Path(args.out_root).resolve()

    info_path = droid_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    fps = int(info.get("fps", 15))

    data_dir = droid_root / "data" / "chunk-000"
    data_files = sorted(data_dir.glob("file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    print(f"[INFO] Found {len(data_files)} DROID parquet files")
    target_eps = None if args.all_episodes else args.num_episodes
    if target_eps is None:
        print("[INFO] Converting all episodes")
    else:
        print(f"[INFO] Converting up to {target_eps} episodes")

    all_episode_meta = []
    all_segment_meta = []
    all_stats = []

    converted = 0
    for source_ep_idx, ep_df in iter_episode_groups(data_files):
        if target_eps is not None and converted >= target_eps:
            break

        new_ep_idx = converted

        # Resume-safe: skip if parquet already exists.
        expected_parquet = out_root / "data" / "droid" / "chunk-000" / f"episode_{new_ep_idx:06d}.parquet"
        if expected_parquet.exists():
            converted += 1
            continue

        ep_meta, seg_meta, stats = convert_episode(
            ep_df=ep_df,
            source_episode_index=source_ep_idx,
            new_episode_index=new_ep_idx,
            droid_root=droid_root,
            out_root=out_root,
            fps=fps,
            copy_video=args.copy_video,
        )
        all_episode_meta.append(ep_meta)
        all_segment_meta.extend(seg_meta)
        all_stats.append(stats)
        converted += 1

        if converted % 1000 == 0:
            print(f"[INFO] Converted {converted} episodes so far")

    if converted == 0:
        raise RuntimeError("No episodes converted from DROID parquet files")

    # If resume skipped existing parquet files, rebuild metadata from converted subset only is not enough.
    # For full conversion runs, recommend using a clean out_root.
    if converted != len(all_episode_meta):
        print("[WARN] Some episodes were skipped because output parquet already existed.")
        print("[WARN] Metadata files now contain only episodes converted in this run.")

    tasks = {}
    unique_tasks = sorted(set(x["task"] for x in all_episode_meta))
    for i, task in enumerate(unique_tasks):
        tasks[str(i)] = {
            "task": task,
            "source_dataset": "DROID",
            "note": "Task text inferred from language_instruction/task_category",
        }

    robots = {
        "franka": {
            "robot_type": "franka",
            "domain": "real",
            "state_dim": int(len(all_stats[0]["state_mean"])),
            "action_dim": int(len(all_stats[0]["action_mean"])),
            "control_mode": "joint",
            "fps": int(fps),
            "cameras": list(CAMERA_MAP.keys()),
            "state_description": "DROID observation.state (joint + gripper)",
            "action_description": "DROID action",
        }
    }

    dataset_stats = {
        "num_episodes": len(all_episode_meta),
        "num_frames": int(sum(x["num_frames"] for x in all_episode_meta)),
        "episodes": all_stats,
    }

    write_jsonl(out_root / "meta" / "episodes.jsonl", all_episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", all_segment_meta)
    write_json(out_root / "meta" / "tasks.json", tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", dataset_stats)
    write_readme(out_root, len(all_episode_meta), fps)

    print("\n[DONE] DROID dataset conversion finished for this run.")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
