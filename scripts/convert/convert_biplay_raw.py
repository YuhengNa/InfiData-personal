import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


CAMERA_KEYS = [
    "cam_high",
    "cam_left_wrist",
    "cam_low",
    "cam_right_wrist",
]

KNOWN_TASK_NAMES = {
    "aloha_pen_uncap_diverse_raw": "uncap the pen",
    "aloha_pen_uncap_diverse_dataset": "uncap the pen",
    "aloha_pick_place_dataset": "pick and place the object",
    "aloha_drawer_dataset": "drawer manipulation",
    "aloha_dough_cut_dataset": "cut the dough",
    "aloha_static_dataset": "static scene manipulation",
    "aloha_play_dataset": "bimanual free-form play",
    "aloha_sushi_cut_full_dataset": "cut and place sushi",
}


def speed_bin_from_T(T, bin_size=500):
    return int(round(T / bin_size) * bin_size)


def find_hdf5_files(root: Path):
    return sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def infer_task_from_root(raw_root: Path, h5_path: Path) -> str:
    for candidate in [raw_root.name, h5_path.parent.name, h5_path.parent.parent.name if len(h5_path.parents) >= 2 else ""]:
        if candidate in KNOWN_TASK_NAMES:
            return KNOWN_TASK_NAMES[candidate]
    task_text = raw_root.name.replace("_raw", "").replace("_dataset", "").replace("aloha_", "")
    return task_text.replace("_", " ").strip() or "biplay manipulation task"


def make_placeholder_segments(T, task):
    cuts = [
        (0, int(T * 0.25), "approach the object"),
        (int(T * 0.25), int(T * 0.50), "grasp or establish contact"),
        (int(T * 0.50), int(T * 0.75), "perform the main manipulation"),
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
    return segments[-1]["subtask"] if segments else "unknown subtask"


def decode_frame(frame_buf: np.ndarray, valid_len):
    if valid_len is not None:
        valid_len = max(0, min(int(valid_len), int(frame_buf.shape[0])))
        encoded = frame_buf[:valid_len].tobytes()
    else:
        encoded = frame_buf.tobytes().rstrip(b"\x00")

    if not encoded:
        return None

    arr = np.frombuffer(encoded, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def infer_compress_len_mapping(f: h5py.File, cameras):
    if "compress_len" not in f:
        return None

    cl = np.asarray(f["compress_len"])
    if cl.ndim != 2:
        return None

    rows, t_len = cl.shape
    cameras = [cam for cam in cameras if cam in f["observations/images"]]
    if rows < len(cameras):
        return None

    score = {}
    for cam in cameras:
        ds = f["observations/images"][cam]
        if ds.ndim != 2 or ds.shape[0] != t_len:
            continue
        idxs = np.linspace(0, ds.shape[0] - 1, num=min(20, ds.shape[0]), dtype=int)
        for row_idx in range(rows):
            ok = 0
            for t in idxs:
                img = decode_frame(np.asarray(ds[t], dtype=np.uint8), int(cl[row_idx, t]))
                if img is not None:
                    ok += 1
            score[(cam, row_idx)] = ok

    if not score:
        return None

    best_score = -1
    best_mapping = None
    for row_perm in itertools.permutations(range(rows), len(cameras)):
        total = sum(score.get((cam, row_idx), 0) for cam, row_idx in zip(cameras, row_perm))
        if total > best_score:
            best_score = total
            best_mapping = dict(zip(cameras, row_perm))
    return best_mapping


def build_video_writer(path: Path, width: int, height: int, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def export_camera_videos(f: h5py.File, out_root: Path, episode_index: int, fps: int):
    image_group = f["observations/images"]
    cameras = [cam for cam in CAMERA_KEYS if cam in image_group]
    mapping = infer_compress_len_mapping(f, cameras)
    compress_len = np.asarray(f["compress_len"]) if "compress_len" in f else None

    video_rel_paths = {}
    for cam in cameras:
        ds = image_group[cam]
        if ds.ndim != 2:
            continue

        out_name = f"episode_{episode_index:06d}_{sanitize_name(cam)}.mp4"
        out_path = out_root / "videos" / "biplay" / out_name
        row_idx = mapping.get(cam) if mapping else None
        lengths = None
        if compress_len is not None:
            if compress_len.ndim == 1 and compress_len.shape[0] == ds.shape[0]:
                lengths = compress_len
            elif compress_len.ndim == 2 and row_idx is not None and row_idx < compress_len.shape[0]:
                lengths = compress_len[row_idx]

        writer = None
        written = 0
        for t in range(ds.shape[0]):
            valid_len = None if lengths is None else int(lengths[t])
            img = decode_frame(np.asarray(ds[t], dtype=np.uint8), valid_len)
            if img is None:
                continue
            if writer is None:
                h, w = img.shape[:2]
                writer = build_video_writer(out_path, w, h, fps)
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer for {out_path}")
            writer.write(img)
            written += 1

        if writer is not None:
            writer.release()

        if written == 0:
            raise RuntimeError(f"no frames decoded for {cam} in episode {episode_index}")

        video_rel_paths[cam] = str(Path("videos") / "biplay" / out_name)

    return video_rel_paths


def convert_one_episode(h5_path: Path, raw_root: Path, out_root: Path, episode_index: int, fps: int, task: str):
    with h5py.File(h5_path, "r") as f:
        qpos = f["observations/qpos"][:].astype(np.float32)
        qvel = f["observations/qvel"][:].astype(np.float32) if "observations/qvel" in f else None
        effort = f["observations/effort"][:].astype(np.float32) if "observations/effort" in f else None
        action = f["action"][:].astype(np.float32)
        option = f["observations/option"][:].astype(np.float32) if "observations/option" in f else None

        T = qpos.shape[0]
        if action.shape[0] != T:
            raise ValueError(f"qpos/action length mismatch in {h5_path}")

        output_video_rel = export_camera_videos(f, out_root, episode_index, fps)

    speed_bin = speed_bin_from_T(T)
    segments = make_placeholder_segments(T, task)

    rows = []
    for t in range(T):
        row = {
            "episode_index": episode_index,
            "frame_index": t,
            "timestamp": float(t / fps),
            "observation.state": qpos[t].tolist(),
            "action": action[t].tolist(),
            "task": task,
            "subtask": subtask_for_frame(t, segments),
            "robot_type": "aloha2_bimanual_viperx",
            "control_mode": "joint",
            "quality": 5,
            "speed_bin": speed_bin,
            "mistake": False,
            "success": True,
            "source_dataset": "BiPlay",
            "domain": "real",
        }

        if qvel is not None:
            row["observation.qvel"] = qvel[t].tolist()
        if effort is not None:
            row["observation.effort"] = effort[t].tolist()
        if option is not None:
            row["observation.option"] = option[t].tolist() if np.ndim(option[t]) > 0 else float(option[t])

        for cam, rel_path in output_video_rel.items():
            row[f"video.{cam}.path"] = rel_path
            row[f"video.{cam}.frame_index"] = t

        subgoal_t = min(t + 2 * fps, T - 1)
        for cam, rel_path in output_video_rel.items():
            row[f"subgoal.real_future.{cam}.path"] = rel_path
            row[f"subgoal.real_future.{cam}.frame_index"] = subgoal_t

        rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path = out_root / "data" / "biplay" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    episode_meta = {
        "episode_index": episode_index,
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
        "source_dataset": "BiPlay",
        "video_paths": output_video_rel,
        "annotation_note": "subtasks are auto-generated placeholders; qpos/action come from raw BiPlay HDF5",
    }

    segment_meta = []
    for i, seg in enumerate(segments):
        item = dict(seg)
        item["episode_index"] = episode_index
        item["segment_index"] = i
        segment_meta.append(item)

    stats = {
        "episode_index": episode_index,
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


def write_readme(out_root: Path, num_eps: int, raw_root: Path):
    text = f"""# biplay_mini

This is a BiPlay prototype dataset converted from raw HDF5 into InfiData format.

## Contents

- Number of converted episodes: {num_eps}
- Source dataset root: {raw_root}
- Robot: ALOHA 2 bimanual ViperX
- Control mode: joint
- FPS: 25
- Cameras: cam_high, cam_left_wrist, cam_low, cam_right_wrist

## Important note

This build only converts raw HDF5 subsets that are locally available.
The current BiPlay workspace contains raw HDF5 for aloha_pen_uncap_diverse_raw; other subsets are present only as TFRecord release packages.
The `subtask` segments are automatically generated placeholders.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--all_episodes", action="store_true")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--task", type=str, default="")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()

    h5_files = find_hdf5_files(raw_root)
    if not h5_files:
        raise FileNotFoundError(f"No hdf5/h5 files found under {raw_root}")

    selected = h5_files if args.all_episodes else h5_files[: args.num_episodes]

    all_episode_meta = []
    all_segment_meta = []
    all_stats = []

    print(f"[INFO] Found {len(h5_files)} HDF5 files")
    print(f"[INFO] Converting {len(selected)} episodes to {out_root}")

    for new_idx, h5_path in enumerate(tqdm(selected, desc="Converting BiPlay")):
        task = args.task.strip() or infer_task_from_root(raw_root, h5_path)
        ep_meta, seg_meta, stats = convert_one_episode(
            h5_path=h5_path,
            raw_root=raw_root,
            out_root=out_root,
            episode_index=new_idx,
            fps=args.fps,
            task=task,
        )
        all_episode_meta.append(ep_meta)
        all_segment_meta.extend(seg_meta)
        all_stats.append(stats)

    write_jsonl(out_root / "meta" / "episodes.jsonl", all_episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", all_segment_meta)

    unique_tasks = sorted(set(x["task"] for x in all_episode_meta))
    tasks = {
        str(i): {
            "task": task_name,
            "source_dataset": "BiPlay",
            "note": "Task inferred from BiPlay subset name when no explicit language instruction is present in raw HDF5.",
        }
        for i, task_name in enumerate(unique_tasks)
    }

    robots = {
        "aloha2_bimanual_viperx": {
            "robot_type": "aloha2_bimanual_viperx",
            "domain": "real",
            "state_dim": 14,
            "action_dim": 14,
            "control_mode": "joint",
            "fps": args.fps,
            "cameras": CAMERA_KEYS,
            "state_description": "14D joint positions, 7 per arm",
            "action_description": "14D action, 7 per arm",
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
    write_readme(out_root, len(all_episode_meta), raw_root)

    print("[DONE] BiPlay dataset conversion finished")


if __name__ == "__main__":
    sys.exit(main())