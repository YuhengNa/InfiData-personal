import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def speed_bin_from_T(T: int, bin_size: int = 500) -> int:
    return int(round(T / bin_size) * bin_size)


def episode_index_from_path(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode") or not stem[len("episode"):].isdigit():
        raise ValueError(f"Unexpected RMBench episode filename: {path.name}")
    return int(stem[len("episode"):])


def decode_image(buf):
    if isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
        arr = buf
    else:
        arr = np.frombuffer(buf, dtype=np.uint8)

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode RMBench image bytes")
    return img


def export_camera_video(rgb_dataset, dst_path: Path, fps: int) -> str:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return str(dst_path)

    first = decode_image(rgb_dataset[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(dst_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {dst_path}")

    writer.write(first)
    for i in range(1, len(rgb_dataset)):
        img = decode_image(rgb_dataset[i])
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        writer.write(img)
    writer.release()
    return str(dst_path)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_task_text(episode_root: Path, source_episode_index: int) -> str:
    instruction_path = episode_root / "instructions" / f"episode{source_episode_index}.json"
    if not instruction_path.exists():
        raise FileNotFoundError(f"Missing instruction file: {instruction_path}")

    instruction = read_json(instruction_path)
    for key in ("seen", "unseen"):
        values = instruction.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip()

    raise ValueError(f"No non-empty instruction text found in {instruction_path}")


def make_segments_from_language_annotation(
    language_annotations,
    source_episode_index: int,
    task: str,
    T: int,
):
    key = f"episode_{source_episode_index}"
    raw_segments = language_annotations.get(key)
    if not raw_segments:
        raise KeyError(f"Missing language annotation for {key}")

    segments = []
    cursor = 0
    for raw_index, raw in enumerate(raw_segments):
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"Malformed language annotation entry for {key}: {raw!r}")

        subtask, frame_count = raw
        if not isinstance(subtask, str) or not subtask.strip():
            raise ValueError(f"Empty subtask text in language annotation for {key}")
        if not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError(f"Invalid frame count in language annotation for {key}: {frame_count!r}")

        start = cursor
        end = min(cursor + frame_count, T) - 1
        cursor += frame_count

        if end < start:
            continue

        segments.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "task": task,
                "subtask": subtask.strip(),
                "annotation_status": "human_labeled",
                "source_annotation_status": "rmbench_language_annotation",
                "source_segment_index": int(raw_index),
                "source_frame_count": int(frame_count),
            }
        )

        if cursor >= T:
            break

    if not segments:
        raise ValueError(f"No usable segments for {key}")
    if segments[-1]["end_frame"] < T - 1:
        raise ValueError(
            f"Language annotation for {key} covers only {segments[-1]['end_frame'] + 1} "
            f"frames, but episode has {T} converted frames"
        )

    return segments


def subtask_for_frame(frame_idx: int, segments) -> str:
    for seg in segments:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return seg["subtask"]
    raise ValueError(f"No segment covers frame {frame_idx}")


def find_task_root(rmbench_root: Path, task_name: str | None, task_config: str) -> tuple[Path, str]:
    root = rmbench_root.resolve()
    if (root / "data").is_dir() and (root / "language_annotation.json").is_file():
        return root, root.parent.name

    if task_name is not None:
        candidate = root / task_name / task_config
        if not candidate.is_dir():
            raise FileNotFoundError(f"RMBench task directory not found: {candidate}")
        return candidate, task_name

    candidates = sorted(p for p in root.glob(f"*/{task_config}") if p.is_dir())
    if len(candidates) == 1:
        return candidates[0], candidates[0].parent.name
    if not candidates:
        raise FileNotFoundError(f"No RMBench task directories matching */{task_config} under {root}")
    raise ValueError(
        "Multiple RMBench tasks found. Pass --task_name explicitly: "
        + ", ".join(p.parent.name for p in candidates)
    )


def list_hdf5_episodes(task_root: Path):
    files = sorted(
        list((task_root / "data").glob("episode*.hdf5")) + list((task_root / "data").glob("episode*.h5")),
        key=episode_index_from_path,
    )
    if not files:
        raise FileNotFoundError(f"No RMBench HDF5 files found under {task_root / 'data'}")
    return files


def convert_one_episode(
    h5_path: Path,
    task_root: Path,
    out_root: Path,
    output_episode_index: int,
    source_episode_index: int,
    task_name: str,
    task_text: str,
    language_annotations,
    fps: int,
    export_videos: bool,
):
    with h5py.File(h5_path, "r") as f:
        vector_path = "/joint_action/vector"
        if vector_path not in f:
            raise KeyError(f"Missing required dataset {vector_path} in {h5_path}")

        joint_vector = f[vector_path][:].astype(np.float32)
        if joint_vector.ndim != 2 or joint_vector.shape[0] < 2:
            raise ValueError(f"Invalid joint vector shape in {h5_path}: {joint_vector.shape}")

        qpos = joint_vector[:-1]
        action = joint_vector[1:]
        T = qpos.shape[0]
        speed_bin = speed_bin_from_T(T)
        segments = make_segments_from_language_annotation(
            language_annotations=language_annotations,
            source_episode_index=source_episode_index,
            task=task_text,
            T=T,
        )

        video_paths = {}
        if export_videos:
            for infi_cam, rmbench_cam in CAMERA_MAP.items():
                rgb_path = f"/observation/{rmbench_cam}/rgb"
                if rgb_path not in f:
                    raise KeyError(f"Missing required camera dataset {rgb_path} in {h5_path}")

                rgb_dataset = f[rgb_path]
                if len(rgb_dataset) < T:
                    raise ValueError(
                        f"Camera {rgb_path} has {len(rgb_dataset)} frames, "
                        f"but episode has {T} converted frames"
                    )

                dst_name = f"episode_{output_episode_index:06d}_{infi_cam}.mp4"
                dst_path = out_root / "videos" / "rmbench" / dst_name
                export_camera_video(rgb_dataset[:T], dst_path, fps=fps)
                video_paths[infi_cam] = str(Path("videos") / "rmbench" / dst_name)

    rows = []
    for t in range(T):
        row = {
            "episode_index": int(output_episode_index),
            "source_episode_index": int(source_episode_index),
            "frame_index": int(t),
            "timestamp": float(t / fps),
            "observation.state": qpos[t].tolist(),
            "action": action[t].tolist(),
            "task": task_text,
            "subtask": subtask_for_frame(t, segments),
            "robot_type": "robotwin_dual_arm_sim",
            "control_mode": "joint",
            "quality": 5,
            "speed_bin": int(speed_bin),
            "mistake": False,
            "success": True,
            "source_dataset": "RMBench",
            "domain": "sim",
            "rmbench_task_name": task_name,
        }

        if export_videos:
            for infi_cam, rel_path in video_paths.items():
                row[f"video.{infi_cam}.path"] = rel_path
                row[f"video.{infi_cam}.frame_index"] = int(t)

            subgoal_t = min(t + 2 * fps, T - 1)
            for infi_cam, rel_path in video_paths.items():
                row[f"subgoal.real_future.{infi_cam}.path"] = rel_path
                row[f"subgoal.real_future.{infi_cam}.frame_index"] = int(subgoal_t)

        rows.append(row)

    out_df = pd.DataFrame(rows)
    parquet_path = out_root / "data" / "rmbench" / "chunk-000" / f"episode_{output_episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(parquet_path, index=False)

    episode_meta = {
        "episode_index": int(output_episode_index),
        "source_episode_index": int(source_episode_index),
        "source_file": str(h5_path),
        "parquet_path": str(parquet_path.relative_to(out_root)),
        "task": task_text,
        "rmbench_task_name": task_name,
        "robot_type": "robotwin_dual_arm_sim",
        "control_mode": "joint",
        "num_frames": int(T),
        "fps": int(fps),
        "duration_sec": float(T / fps),
        "speed_bin": int(speed_bin),
        "quality": 5,
        "success": True,
        "mistake": False,
        "source_dataset": "RMBench",
        "domain": "sim",
        "video_paths": video_paths,
        "annotation_note": "Subtasks are converted from RMBench language_annotation.json; no placeholder subtasks are generated.",
    }

    segment_meta = []
    for segment_index, seg in enumerate(segments):
        item = dict(seg)
        item["episode_index"] = int(output_episode_index)
        item["segment_index"] = int(segment_index)
        item["source_episode_index"] = int(source_episode_index)
        segment_meta.append(item)

    stats = {
        "episode_index": int(output_episode_index),
        "source_episode_index": int(source_episode_index),
        "num_frames": int(T),
        "state_min": qpos.min(axis=0).tolist(),
        "state_max": qpos.max(axis=0).tolist(),
        "state_mean": qpos.mean(axis=0).tolist(),
        "state_std": qpos.std(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": action.std(axis=0).tolist(),
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


def write_readme(out_root: Path, num_eps: int, task_name: str, fps: int, export_videos: bool):
    video_note = (
        "- `video.cam_*.path` and `video.cam_*.frame_index` point to exported mp4 files."
        if export_videos
        else "- Video columns are omitted because `--no_export_videos` was used."
    )
    text = f"""# rmbench_mini

This is an InfiData-style prototype dataset converted from RMBench.

## Contents

- Number of converted episodes: {num_eps}
- Source dataset: RMBench
- RMBench task: {task_name}
- RMBench config: demo_clean
- Robot: RoboTwin dual-arm simulation
- Control mode: joint
- FPS: {fps}
- Cameras: cam_high, cam_left_wrist, cam_right_wrist

## Field notes

- `observation.state` is RMBench `/joint_action/vector` at timestep `t`.
- `action` is the next-step RMBench `/joint_action/vector` at timestep `t + 1`.
- `task` is read from RMBench `instructions/episode*.json`.
- `subtask` and `meta/segments.jsonl` are converted from RMBench `language_annotation.json`.
- `subgoal.real_future.*` is a future-frame pointer, not a human semantic goal label.
{video_note}

## Important notes

This converter does not create placeholder task, subtask, state, action, or camera data.
Episodes with missing required RMBench files or inconsistent frame counts are skipped by default, or fail immediately with `--strict`.
Subtasks are derived from `language_annotation.json`, and segment records use `annotation_status=human_labeled` with `source_annotation_status=rmbench_language_annotation`.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="examples/rmbench_mini")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no_export_videos", action="store_true")
    parser.add_argument("--strict", action="store_true", help="fail on the first invalid episode instead of skipping it")
    args = parser.parse_args()

    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive")

    rmbench_root = Path(args.rmbench_root)
    out_root = Path(args.out_root).resolve()
    task_root, task_name = find_task_root(rmbench_root, args.task_name, args.task_config)
    language_annotation_path = task_root / "language_annotation.json"
    if not language_annotation_path.exists():
        raise FileNotFoundError(f"Missing RMBench language annotation file: {language_annotation_path}")

    language_annotations = read_json(language_annotation_path)
    h5_files = list_hdf5_episodes(task_root)
    export_videos = not args.no_export_videos

    print(f"[INFO] RMBench task root: {task_root}")
    print(f"[INFO] Found {len(h5_files)} HDF5 episodes")
    print(f"[INFO] Converting up to {args.num_episodes} valid episodes")
    print(f"[INFO] Output root: {out_root}")

    all_episode_meta = []
    all_segment_meta = []
    all_stats = []
    skipped = []

    for h5_path in tqdm(h5_files, desc="Converting RMBench episodes"):
        if len(all_episode_meta) >= args.num_episodes:
            break

        source_episode_index = episode_index_from_path(h5_path)
        try:
            task_text = load_task_text(task_root, source_episode_index)
            ep_meta, seg_meta, stats = convert_one_episode(
                h5_path=h5_path,
                task_root=task_root,
                out_root=out_root,
                output_episode_index=len(all_episode_meta),
                source_episode_index=source_episode_index,
                task_name=task_name,
                task_text=task_text,
                language_annotations=language_annotations,
                fps=args.fps,
                export_videos=export_videos,
            )
        except Exception as exc:
            if args.strict:
                raise
            skipped.append({"source_file": str(h5_path), "reason": str(exc)})
            print(f"[WARN] Skipping {h5_path}: {exc}")
            continue

        all_episode_meta.append(ep_meta)
        all_segment_meta.extend(seg_meta)
        all_stats.append(stats)

    if not all_episode_meta:
        raise RuntimeError("No valid RMBench episodes were converted")

    tasks = {
        str(i): {
            "task": task,
            "source_dataset": "RMBench",
            "rmbench_task_name": task_name,
        }
        for i, task in enumerate(sorted(set(x["task"] for x in all_episode_meta)))
    }

    state_dim = int(len(all_stats[0]["state_mean"]))
    action_dim = int(len(all_stats[0]["action_mean"]))
    robots = {
        "robotwin_dual_arm_sim": {
            "robot_type": "robotwin_dual_arm_sim",
            "domain": "sim",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "control_mode": "joint",
            "fps": int(args.fps),
            "cameras": list(CAMERA_MAP.keys()) if export_videos else [],
            "state_description": "RMBench /joint_action/vector at timestep t",
            "action_description": "RMBench next-step /joint_action/vector at timestep t+1",
        }
    }

    dataset_stats = {
        "num_episodes": len(all_episode_meta),
        "num_frames": int(sum(x["num_frames"] for x in all_episode_meta)),
        "episodes": all_stats,
        "skipped_episodes": skipped,
    }

    write_jsonl(out_root / "meta" / "episodes.jsonl", all_episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", all_segment_meta)
    write_json(out_root / "meta" / "tasks.json", tasks)
    write_json(out_root / "meta" / "robots.json", robots)
    write_json(out_root / "meta" / "stats.json", dataset_stats)
    write_readme(out_root, len(all_episode_meta), task_name=task_name, fps=args.fps, export_videos=export_videos)

    print("\n[DONE] RMBench mini dataset built.")
    print(f"Output: {out_root}")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} incomplete or invalid episodes. See meta/stats.json.")


if __name__ == "__main__":
    main()
