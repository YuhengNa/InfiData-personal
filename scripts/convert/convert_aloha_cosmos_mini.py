import argparse
import json
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


CAMERA_KEYS = [
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]


def decode_h5_string(x):
    """Decode HDF5 scalar string."""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray):
        x = x.item()
        if isinstance(x, bytes):
            return x.decode("utf-8")
        return str(x)
    return str(x)


def speed_bin_from_T(T, bin_size=500):
    """π0.7-style coarse episode length bin."""
    return int(round(T / bin_size) * bin_size)


def safe_symlink_or_copy(src: Path, dst: Path, copy_video=False):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return

    if not src.exists():
        print(f"[WARN] video not found: {src}")
        return

    if copy_video:
        shutil.copy2(src, dst)
    else:
        # symlink saves disk space
        os.symlink(src.resolve(), dst)


def find_hdf5_files(root: Path):
    files = sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))
    return files


def infer_task_from_h5_path(h5_path: Path, aloha_root: Path):
    """Infer canonical task text from preprocessed dataset folder name."""
    try:
        rel = h5_path.resolve().relative_to(aloha_root.resolve())
        # Expected: <task_folder>/train/episode_x.hdf5
        if len(rel.parts) >= 3:
            task_folder = rel.parts[0]
        else:
            task_folder = h5_path.parents[1].name
    except Exception:
        task_folder = h5_path.parents[1].name

    # Remove split/count suffixes such as _demos_1000_steps_each_pt1
    task_core = re.sub(r"_demos_.*$", "", task_folder)
    task_text = task_core.replace("_", " ").strip()
    return task_text if task_text else "aloha manipulation task"


def read_video_paths(f, h5_path: Path):
    video_paths = {}

    for cam in CAMERA_KEYS:
        key = f"observations/video_paths/{cam}"
        if key in f:
            rel = decode_h5_string(f[key][()])
            candidate = (h5_path.parent / rel).resolve()
            video_paths[cam] = candidate
        else:
            video_paths[cam] = None

    return video_paths


def make_placeholder_segments(T, task):
    """
    先生成占位 subtask。
    注意：这是 demo 版，不是真实人工语义标注。
    后面可以人工或用 VLM 修正。
    """
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


def subtask_for_frame(frame_idx, segments):
    for seg in segments:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return seg["subtask"]
    return segments[-1]["subtask"]


def convert_one_episode(
    h5_path: Path,
    out_root: Path,
    new_episode_index: int,
    task: str,
    copy_video: bool = False,
):
    with h5py.File(h5_path, "r") as f:
        qpos = f["observations/qpos"][:].astype(np.float32)
        action = f["action"][:].astype(np.float32)

        qvel = f["observations/qvel"][:].astype(np.float32) if "observations/qvel" in f else None
        effort = f["observations/effort"][:].astype(np.float32) if "observations/effort" in f else None
        rel_action = f["relative_action"][:].astype(np.float32) if "relative_action" in f else None

        T = qpos.shape[0]
        assert action.shape[0] == T, f"qpos/action length mismatch in {h5_path}"

        video_src_paths = read_video_paths(f, h5_path)

    fps = 25
    speed_bin = speed_bin_from_T(T)

    # 输出视频路径
    output_video_rel = {}
    for cam, src_path in video_src_paths.items():
        if src_path is None:
            output_video_rel[cam] = None
            continue

        dst_name = f"episode_{new_episode_index:06d}_{cam}.mp4"
        dst_path = out_root / "videos" / "aloha" / dst_name
        safe_symlink_or_copy(src_path, dst_path, copy_video=copy_video)
        output_video_rel[cam] = str(Path("videos") / "aloha" / dst_name)

    segments = make_placeholder_segments(T, task)

    rows = []
    for t in range(T):
        row = {
            "episode_index": new_episode_index,
            "frame_index": t,
            "timestamp": float(t / fps),

            # robot state/action
            "observation.state": qpos[t].tolist(),
            "action": action[t].tolist(),

            # prompt/context fields
            "task": task,
            "subtask": subtask_for_frame(t, segments),
            "robot_type": "aloha2_bimanual_viperx",
            "control_mode": "joint",
            "quality": 5,
            "speed_bin": speed_bin,
            "mistake": False,
            "success": True,
            "source_dataset": "ALOHA-Cosmos-Policy",
        }

        if qvel is not None:
            row["observation.qvel"] = qvel[t].tolist()
        if effort is not None:
            row["observation.effort"] = effort[t].tolist()
        if rel_action is not None:
            row["action_relative"] = rel_action[t].tolist()

        for cam in CAMERA_KEYS:
            row[f"video.{cam}.path"] = output_video_rel.get(cam)
            row[f"video.{cam}.frame_index"] = t

        # demo 版 subgoal：取 2 秒后的未来帧，超过末尾则取最后一帧
        subgoal_t = min(t + 2 * fps, T - 1)
        for cam in CAMERA_KEYS:
            row[f"subgoal.real_future.{cam}.path"] = output_video_rel.get(cam)
            row[f"subgoal.real_future.{cam}.frame_index"] = subgoal_t

        rows.append(row)

    df = pd.DataFrame(rows)

    parquet_path = (
        out_root
        / "data"
        / "aloha"
        / "chunk-000"
        / f"episode_{new_episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    episode_meta = {
        "episode_index": new_episode_index,
        "source_file": str(h5_path),
        "parquet_path": str(parquet_path.relative_to(out_root)),
        "task": task,
        "robot_type": "aloha2_bimanual_viperx",
        "control_mode": "joint",
        "num_frames": T,
        "fps": fps,
        "duration_sec": T / fps,
        "speed_bin": speed_bin,
        "quality": 5,
        "success": True,
        "mistake": False,
        "source_dataset": "ALOHA-Cosmos-Policy",
        "video_paths": output_video_rel,
        "annotation_note": "quality/success/mistake initialized from successful-demo assumption; subtasks are placeholders",
    }

    segment_meta = []
    for i, seg in enumerate(segments):
        item = dict(seg)
        item["episode_index"] = new_episode_index
        item["segment_index"] = i
        segment_meta.append(item)

    stats = {
        "episode_index": new_episode_index,
        "num_frames": T,
        "qpos_min": qpos.min(axis=0).tolist(),
        "qpos_max": qpos.max(axis=0).tolist(),
        "qpos_mean": qpos.mean(axis=0).tolist(),
        "qpos_std": qpos.std(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": action.std(axis=0).tolist(),
    }

    return episode_meta, segment_meta, stats


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_readme(out_root: Path, num_eps: int):
    text = f"""# my_pi07_dataset_mini

This is a small π0.7-style prototype dataset converted from ALOHA-Cosmos-Policy.

## Contents

- Number of converted episodes: {num_eps}
- Source dataset: ALOHA-Cosmos-Policy
- Robot: ALOHA 2 bimanual ViperX
- Control mode: joint
- FPS: 25
- Cameras: cam_high, cam_left_wrist, cam_right_wrist

## Important note

This is a prototype for presentation and data pipeline validation.

The fields `quality`, `success`, and `mistake` are initialized using the assumption that ALOHA-Cosmos-Policy contains successful demonstrations.
The `subtask` segments are automatically generated placeholders and should be replaced by human or VLM-assisted annotations before real training.

## π0.7-style fields

Each parquet row corresponds to one timestep and includes:

- observation.state
- observation.qvel
- observation.effort
- action
- action_relative
- task
- subtask
- robot_type
- control_mode
- quality
- speed_bin
- mistake
- success
- video paths and frame indices
- real-future subgoal frame pointers
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aloha_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="my_pi07_dataset_mini")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--all_episodes", action="store_true", help="convert all episodes under aloha_root")
    parser.add_argument("--task", type=str, default="aloha manipulation task")
    parser.add_argument("--task_mode", type=str, default="auto", choices=["auto", "fixed"], help="auto: infer task from folder name; fixed: use --task for all episodes")
    parser.add_argument("--copy_video", action="store_true", help="copy videos instead of symlink")
    args = parser.parse_args()

    aloha_root = Path(args.aloha_root).resolve()
    out_root = Path(args.out_root).resolve()

    h5_files = find_hdf5_files(aloha_root)
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No hdf5/h5 files found under {aloha_root}")

    selected = h5_files if args.all_episodes else h5_files[: args.num_episodes]

    print(f"[INFO] Found {len(h5_files)} HDF5 files")
    print(f"[INFO] Converting {len(selected)} episodes")
    print(f"[INFO] Output root: {out_root}")

    all_episode_meta = []
    all_segment_meta = []
    all_stats = []

    for new_idx, h5_path in enumerate(tqdm(selected)):
        task_text = args.task if args.task_mode == "fixed" else infer_task_from_h5_path(h5_path, aloha_root)
        ep_meta, seg_meta, stats = convert_one_episode(
            h5_path=h5_path,
            out_root=out_root,
            new_episode_index=new_idx,
            task=task_text,
            copy_video=args.copy_video,
        )
        all_episode_meta.append(ep_meta)
        all_segment_meta.extend(seg_meta)
        all_stats.append(stats)

    write_jsonl(out_root / "meta" / "episodes.jsonl", all_episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", all_segment_meta)

    unique_tasks = sorted(set(x["task"] for x in all_episode_meta))
    tasks = {}
    for i, task_name in enumerate(unique_tasks):
        tasks[str(i)] = {
            "task": task_name,
            "source_dataset": "ALOHA-Cosmos-Policy",
            "note": "Task inferred from source folder name when task_mode=auto.",
        }

    robots = {
        "aloha2_bimanual_viperx": {
            "robot_type": "aloha2_bimanual_viperx",
            "domain": "real",
            "state_dim": 14,
            "action_dim": 14,
            "control_mode": "joint",
            "fps": 25,
            "cameras": CAMERA_KEYS,
            "state_description": "14D joint positions, 7 per arm",
            "action_description": "14D absolute joint-position action, 7 per arm",
        }
    }

    dataset_stats = {
        "num_episodes": len(all_episode_meta),
        "num_frames": int(sum(x["num_frames"] for x in all_episode_meta)),
        "episodes": all_stats,
    }

    write_json(out_root / "meta" / "tasks.json", tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", dataset_stats)
    write_readme(out_root, len(all_episode_meta))

    print("\n[DONE] Mini dataset built.")
    print(f"Output: {out_root}")
    print("Try:")
    print(f"  find {out_root} -maxdepth 4 -type f | head -50")


if __name__ == "__main__":
    main()