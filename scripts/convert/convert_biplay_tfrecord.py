import argparse
import json
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tfrecord.reader import tfrecord_loader
from tqdm import tqdm


CAMERA_KEYS = [
    "cam_high",
    "cam_left_wrist",
    "cam_low",
    "cam_right_wrist",
]

KNOWN_TASK_NAMES = {
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


def find_tfrecord_files(dataset_root: Path):
    version_dir = dataset_root / "1.0.0"
    if not version_dir.exists():
        raise FileNotFoundError(f"Missing TFDS version directory: {version_dir}")
    return sorted(version_dir.glob("*.tfrecord-*"))


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def decode_text(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    if isinstance(x, np.bytes_):
        return bytes(x).decode("utf-8", errors="ignore").strip()
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def first_non_empty_text(*arrays) -> str:
    for arr in arrays:
        if arr is None:
            continue
        for x in arr:
            text = decode_text(x)
            if text:
                return text
    return ""


def infer_task(dataset_root: Path, record: dict) -> str:
    task = first_non_empty_text(
        record.get("steps/language_instruction"),
        record.get("steps/global_instruction"),
        record.get("steps/clip_instruction"),
    )
    if task:
        return task
    return KNOWN_TASK_NAMES.get(dataset_root.name, dataset_root.name.replace("_dataset", "").replace("aloha_", "").replace("_", " "))


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


def build_video_writer(path: Path, width: int, height: int, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def decode_image(image_bytes):
    encoded = bytes(image_bytes)
    if not encoded:
        return None
    return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)


def reshape_trajectory(arr, T, dim, key):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size != T * dim:
        raise ValueError(f"Unexpected shape for {key}: flat size={arr.size}, expected={T * dim}")
    return arr.reshape(T, dim)


def export_videos(record: dict, out_root: Path, episode_index: int, fps: int, T: int):
    rel_paths = {}
    for cam in CAMERA_KEYS:
        key = f"steps/observation/{cam}"
        if key not in record:
            rel_paths[cam] = None
            continue

        cam_frames = record[key]
        if getattr(cam_frames, "size", 0) == 0 or len(cam_frames) == 0:
            rel_paths[cam] = None
            continue

        out_name = f"episode_{episode_index:06d}_{sanitize_name(cam)}.mp4"
        out_path = out_root / "videos" / "biplay" / out_name
        writer = None
        written = 0
        for t in range(min(T, len(cam_frames))):
            img = decode_image(cam_frames[t])
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
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            rel_paths[cam] = None
        else:
            rel_paths[cam] = str(Path("videos") / "biplay" / out_name)

    return rel_paths


def convert_record(record: dict, dataset_root: Path, out_root: Path, episode_index: int, fps: int):
    T = int(len(record["steps/is_first"]))
    if T == 0:
        raise ValueError(f"empty trajectory at episode {episode_index}")

    qpos = reshape_trajectory(record["steps/observation/state"], T, 14, "steps/observation/state")
    action = reshape_trajectory(record["steps/action"], T, 14, "steps/action")
    reward = np.asarray(record.get("steps/reward", np.zeros(T, dtype=np.float32)), dtype=np.float32)
    discount = np.asarray(record.get("steps/discount", np.ones(T, dtype=np.float32)), dtype=np.float32)
    is_terminal = np.asarray(record.get("steps/is_terminal", np.zeros(T, dtype=bool)), dtype=bool)
    video_rel_paths = export_videos(record, out_root, episode_index, fps, T)

    task = infer_task(dataset_root, record)
    speed_bin = speed_bin_from_T(T)
    success = bool(is_terminal[-1]) if len(is_terminal) else True
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
            "quality": 5 if success else 2,
            "speed_bin": speed_bin,
            "mistake": not success,
            "success": success,
            "source_dataset": "BiPlay",
            "domain": "real",
            "reward": float(reward[t]),
            "discount": float(discount[t]),
        }

        for cam, rel_path in video_rel_paths.items():
            row[f"video.{cam}.path"] = rel_path
            row[f"video.{cam}.frame_index"] = t

        subgoal_t = min(t + 2 * fps, T - 1)
        for cam, rel_path in video_rel_paths.items():
            row[f"subgoal.real_future.{cam}.path"] = rel_path
            row[f"subgoal.real_future.{cam}.frame_index"] = subgoal_t

        rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path = out_root / "data" / "biplay" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    file_path = decode_text(record.get("episode_metadata/file_path", b""))
    episode_meta = {
        "episode_index": episode_index,
        "source_file": file_path,
        "parquet_path": str(parquet_path.relative_to(out_root)),
        "task": task,
        "robot_type": "aloha2_bimanual_viperx",
        "control_mode": "joint",
        "num_frames": T,
        "fps": fps,
        "duration_sec": T / fps,
        "speed_bin": speed_bin,
        "quality": 5 if success else 2,
        "success": success,
        "mistake": not success,
        "source_dataset": "BiPlay",
        "video_paths": video_rel_paths,
        "annotation_note": "Converted from BiPlay TFRecord release package; subtasks are auto-generated placeholders",
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


def iter_records(dataset_root: Path):
    for tfrecord_path in find_tfrecord_files(dataset_root):
        for record in tfrecord_loader(str(tfrecord_path), None, None):
            yield record


def count_existing_parquets(out_root: Path) -> int:
    parquet_dir = out_root / "data" / "biplay" / "chunk-000"
    if not parquet_dir.exists():
        return 0
    return len(list(parquet_dir.glob("episode_*.parquet")))


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_readme(out_root: Path, dataset_root: Path, num_eps: int):
    text = f"""# biplay_tfrecord

This dataset was converted from the BiPlay TFRecord release package into InfiData format.

## Contents

- Number of converted episodes: {num_eps}
- Source dataset root: {dataset_root}
- Robot: ALOHA 2 bimanual ViperX
- Control mode: joint
- FPS: 25
- Cameras: cam_high, cam_left_wrist, cam_low, cam_right_wrist

## Important note

This build consumes the published TFRecord package, not the original raw HDF5.
The `subtask` segments are automatically generated placeholders.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--all_episodes", action="store_true")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.out_root).resolve()

    start_index = count_existing_parquets(out_root)
    if start_index:
        print(f"[INFO] Resuming from existing episode index {start_index}")

    records_iter = itertools.islice(iter_records(dataset_root), start_index, None)
    selected = records_iter if args.all_episodes else (record for _, record in zip(range(args.num_episodes), records_iter))

    all_episode_meta = []
    all_segment_meta = []
    all_stats = []

    converted = 0
    skipped = 0
    for offset, record in enumerate(tqdm(selected, desc=f"Converting {dataset_root.name}")):
        episode_index = start_index + offset
        try:
            ep_meta, seg_meta, stats = convert_record(record, dataset_root, out_root, episode_index, args.fps)
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skip episode {episode_index}: {exc}")
            continue
        all_episode_meta.append(ep_meta)
        all_segment_meta.extend(seg_meta)
        all_stats.append(stats)
        converted += 1

    if converted == 0:
        raise RuntimeError(f"No TFRecord episodes were converted from {dataset_root}")

    write_jsonl(out_root / "meta" / "episodes.jsonl", all_episode_meta)
    write_jsonl(out_root / "meta" / "segments.jsonl", all_segment_meta)

    unique_tasks = sorted(set(x["task"] for x in all_episode_meta))
    tasks = {
        str(i): {
            "task": task_name,
            "source_dataset": "BiPlay",
            "note": "Task inferred from TFRecord language fields or dataset name.",
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
    write_readme(out_root, dataset_root, len(all_episode_meta))
    print(f"[DONE] Converted {len(all_episode_meta)} episodes from {dataset_root.name} (skipped={skipped})")


if __name__ == "__main__":
    sys.exit(main())