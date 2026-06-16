import argparse
import io
import json
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from convert_common import (
    episode_chunk,
    speed_bin_from_T,
    vector_stats,
    write_json,
    write_jsonl,
)


CAMERA_MAP = {
    "cam_high": "observations/rgb_images/camera_front",
    "cam_left_wrist": "observations/rgb_images/camera_left_wrist",
    "cam_right_wrist": "observations/rgb_images/camera_right_wrist",
}


class MultiPartReader(io.RawIOBase):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.index = 0
        self.current = None
        self._open_next()

    def _open_next(self) -> None:
        if self.current is not None:
            self.current.close()
        if self.index >= len(self.paths):
            self.current = None
            return
        self.current = self.paths[self.index].open("rb")
        self.index += 1

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self.current is None:
            return 0
        size = self.current.readinto(buffer)
        if size:
            return size
        self._open_next()
        return self.readinto(buffer)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
        super().close()


def multipart_groups(root: Path) -> list[tuple[str, list[Path], Path]]:
    groups = defaultdict(list)
    for path in sorted(root.rglob("*.tar.gz.part-*")):
        if re.fullmatch(r".+\.tar\.gz\.part-[a-z]+", path.name) is None:
            continue
        stem = path.name.split(".tar.gz.part-")[0]
        groups[(path.parent, stem)].append(path)

    ready = []
    for (parts_dir, stem), paths in groups.items():
        paths = sorted(paths, key=lambda path: path.name)
        relative_parent = parts_dir.relative_to(root)
        ready.append((stem, paths, relative_parent))
    return ready


def safe_member_name(name: str) -> Path | None:
    cleaned = name.lstrip("/").replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    path = Path(cleaned)
    if not cleaned or ".." in path.parts:
        return None
    return path


def extraction_target(extract_root: Path, relative_parent: Path) -> Path:
    parts = list(relative_parent.parts)
    if parts and parts[0].endswith("_compressed"):
        parts[0] = parts[0].replace("_compressed", "_release")
    return extract_root.joinpath(*parts)


def extract_hdf5_archives(robomind_root: Path, extract_root: Path) -> None:
    groups = multipart_groups(robomind_root)
    if not groups:
        raise FileNotFoundError(f"No multipart RoboMIND archives found under {robomind_root}")

    for stem, paths, relative_parent in tqdm(groups, desc="Extracting RoboMIND HDF5 archives"):
        target_root = extraction_target(extract_root, relative_parent)
        marker = target_root / ".extract_markers" / f"{stem}.ok"
        if marker.exists():
            continue

        target_root.mkdir(parents=True, exist_ok=True)
        reader = MultiPartReader(paths)
        extracted = 0
        try:
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith("trajectory.hdf5"):
                        continue
                    relative = safe_member_name(member.name)
                    if relative is None:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    destination = target_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as output:
                        shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
                    extracted += 1
        finally:
            reader.close()

        if extracted == 0:
            raise RuntimeError(f"No trajectory.hdf5 found in archive group {stem}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{extracted}\n", encoding="utf-8")


def discover_hdf5_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("trajectory.hdf5"))
    if not files:
        files = sorted(root.rglob("*.hdf5"))
    return files


def find_annotation_json(root: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"RoboMIND annotation JSON not found: {path}")
        return path

    candidates = sorted(
        root.rglob("language_description_annotation_json/h5_agilex_3rgb.json")
    )
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_annotations(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record["id"]).strip("/"): record.get("response", {})
        for record in records
        if isinstance(record, dict) and record.get("id")
    }


def annotation_id_from_path(path: Path) -> str | None:
    parts = list(path.parts)
    embodiment_index = None
    for index, part in enumerate(parts):
        if part.startswith("h5_"):
            embodiment_index = index
    if embodiment_index is None:
        return None
    return "/".join(parts[embodiment_index:-1]).strip("/")


def decode_h5_text(value) -> str:
    array = np.asarray(value).reshape(-1)
    for item in array:
        if isinstance(item, bytes):
            text = item.decode("utf-8", errors="ignore").strip()
        else:
            text = str(item).strip()
        if text:
            return text
    return ""


def read_joint_data(h5_file: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    required = [
        "puppet/joint_position_left",
        "puppet/joint_position_right",
        "master/joint_position_left",
        "master/joint_position_right",
    ]
    missing = [name for name in required if name not in h5_file]
    if missing:
        raise KeyError(f"Unsupported RoboMIND HDF5 schema; missing {missing}")

    state = np.concatenate(
        [
            h5_file["puppet/joint_position_left"][:].astype(np.float32),
            h5_file["puppet/joint_position_right"][:].astype(np.float32),
        ],
        axis=1,
    )
    action = np.concatenate(
        [
            h5_file["master/joint_position_left"][:].astype(np.float32),
            h5_file["master/joint_position_right"][:].astype(np.float32),
        ],
        axis=1,
    )
    if state.shape != action.shape:
        raise ValueError(f"RoboMIND state/action shape mismatch: {state.shape} vs {action.shape}")
    return state, action


def frame_number(value) -> int:
    match = re.search(r"_(\d+)\.[A-Za-z0-9]+$", str(value))
    if not match:
        raise ValueError(f"Cannot parse RoboMIND frame name: {value}")
    return int(match.group(1))


def make_segments(
    annotation: dict,
    task: str,
    T: int,
    output_episode_index: int,
    source_file: Path,
) -> list[dict]:
    raw_steps = annotation.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return [
            {
                "episode_index": int(output_episode_index),
                "segment_index": 0,
                "start_frame": 0,
                "end_frame": int(T - 1),
                "task": task,
                "subtask": task,
                "annotation_status": "auto_placeholder",
                "source_file": str(source_file),
            }
        ]

    parsed = []
    for raw in raw_steps:
        description = str(raw.get("step_description", "")).strip()
        if not description:
            continue
        start = frame_number(raw.get("start_frame"))
        end = frame_number(raw.get("end_frame"))
        parsed.append((start, end, description))
    parsed.sort(key=lambda item: item[0])
    if not parsed:
        return [
            {
                "episode_index": int(output_episode_index),
                "segment_index": 0,
                "start_frame": 0,
                "end_frame": int(T - 1),
                "task": task,
                "subtask": task,
                "annotation_status": "auto_placeholder",
                "source_file": str(source_file),
            }
        ]

    segments = []
    for index, (source_start, source_end, description) in enumerate(parsed):
        start = 0 if index == 0 else min(max(source_start, 0), T - 1)
        if index + 1 < len(parsed):
            end = min(max(parsed[index + 1][0] - 1, start), T - 1)
        else:
            end = T - 1
        segments.append(
            {
                "episode_index": int(output_episode_index),
                "segment_index": len(segments),
                "start_frame": int(start),
                "end_frame": int(end),
                "task": task,
                "subtask": description,
                "annotation_status": "human_labeled",
                "source_annotation_status": "robomind_language_description_annotation",
                "source_start_frame": int(source_start),
                "source_end_frame": int(source_end),
                "source_file": str(source_file),
            }
        )
    return segments


def subtask_for_frame(frame_index: int, segments: list[dict]) -> str:
    for segment in segments:
        if segment["start_frame"] <= frame_index <= segment["end_frame"]:
            return segment["subtask"]
    return segments[-1]["subtask"]


def decode_jpeg(value) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.ndim == 3:
        image = value
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        raise ValueError(f"Unsupported raw RoboMIND image shape: {image.shape}")

    raw = np.asarray(value, dtype=np.uint8).reshape(-1)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode RoboMIND JPEG frame")
    return image


def export_camera_video(dataset, destination: Path, fps: int, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    if destination.exists():
        destination.unlink()

    first = decode_jpeg(dataset[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {destination}")

    try:
        writer.write(first)
        for index in range(1, len(dataset)):
            image = decode_jpeg(dataset[index])
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
    finally:
        writer.release()


def convert_episode(
    h5_path: Path,
    annotation_lookup: dict[str, dict],
    output_episode_index: int,
    out_root: Path,
    fps: int,
    no_videos: bool,
    overwrite: bool,
    future_seconds: float,
):
    annotation_id = annotation_id_from_path(h5_path)
    annotation = annotation_lookup.get(annotation_id or "", {})

    with h5py.File(h5_path, "r") as h5_file:
        state, action = read_joint_data(h5_file)
        T = len(state)
        if T == 0:
            raise ValueError(f"Empty RoboMIND episode: {h5_path}")

        task = ""
        if "language_raw" in h5_file:
            task = decode_h5_text(h5_file["language_raw"][()])
        if not task:
            task = str(annotation.get("task_summary", "")).strip()
        if not task:
            task = h5_path.parents[4].name.replace("_", " ")

        segments = make_segments(
            annotation=annotation,
            task=task,
            T=T,
            output_episode_index=output_episode_index,
            source_file=h5_path,
        )

        video_paths = {}
        if not no_videos:
            for infi_camera, source_camera in CAMERA_MAP.items():
                if source_camera not in h5_file:
                    raise KeyError(f"Missing RoboMIND camera {source_camera} in {h5_path}")
                if len(h5_file[source_camera]) != T:
                    raise ValueError(
                        f"RoboMIND camera length mismatch in {h5_path}: "
                        f"{source_camera} has {len(h5_file[source_camera])}, expected {T}"
                    )

                rel = (
                    Path("videos")
                    / "robomind"
                    / infi_camera
                    / f"episode_{output_episode_index:06d}.mp4"
                )
                export_camera_video(
                    h5_file[source_camera],
                    out_root / rel,
                    fps=fps,
                    overwrite=overwrite,
                )
                video_paths[infi_camera] = str(rel)

        is_success = "success_episodes" in h5_path.parts
        is_failure = "failure_episodes" in h5_path.parts or "fail_episodes" in h5_path.parts
        success = is_success or not is_failure
        mistake = not success
        quality = 5 if success else 2
        outcome_status = "source_path_label" if is_success or is_failure else "auto_placeholder"

    speed_bin = speed_bin_from_T(T)
    future_offset = max(1, int(round(future_seconds * fps)))
    annotation_status = segments[0]["annotation_status"]
    rows = []
    for t in range(T):
        row = {
            "episode_index": int(output_episode_index),
            "frame_index": int(t),
            "timestamp": float(t / fps),
            "observation.state": state[t].tolist(),
            "action": action[t].tolist(),
            "task": task,
            "subtask": subtask_for_frame(t, segments),
            "robot_type": "agilex_cobot_magic",
            "control_mode": "joint",
            "quality": int(quality),
            "speed_bin": int(speed_bin),
            "mistake": bool(mistake),
            "success": bool(success),
            "source_dataset": "RoboMIND",
            "domain": "real",
            "subtask_annotation_status": annotation_status,
            "outcome_annotation_status": outcome_status,
        }
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
    parquet_path = (
        out_root
        / "data"
        / "robomind"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{output_episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists() and not overwrite:
        raise FileExistsError(f"Output parquet already exists: {parquet_path}")
    out_df.to_parquet(parquet_path, index=False)

    episode_meta = {
        "episode_index": int(output_episode_index),
        "source_file": str(h5_path),
        "source_annotation_id": annotation_id,
        "parquet_path": str(parquet_path.relative_to(out_root)),
        "task": task,
        "robot_type": "agilex_cobot_magic",
        "control_mode": "joint",
        "num_frames": int(T),
        "fps": int(fps),
        "duration_sec": float(T / fps),
        "speed_bin": int(speed_bin),
        "quality": int(quality),
        "success": bool(success),
        "mistake": bool(mistake),
        "source_dataset": "RoboMIND",
        "domain": "real",
        "video_paths": video_paths,
        "camera_mapping": CAMERA_MAP if not no_videos else {},
        "subtask_annotation_status": annotation_status,
        "outcome_annotation_status": outcome_status,
        "annotation_note": (
            "State uses puppet joint positions; action uses master joint positions. "
            "Subtasks use official RoboMIND language-description steps when available."
        ),
    }
    stats = vector_stats(state, action)
    stats["episode_index"] = int(output_episode_index)
    return episode_meta, segments, stats


def write_readme(
    out_root: Path,
    num_episodes: int,
    fps: int,
    annotation_path: Path | None,
    no_videos: bool,
) -> None:
    text = f"""# RoboMIND InfiData

Converted from RoboMIND Agilex Cobot Magic HDF5 trajectories.

- Episodes: {num_episodes}
- Robot: Agilex Cobot Magic
- Domain: real
- FPS: {fps}
- State: puppet left/right joint positions
- Action: master left/right joint positions
- Videos included: {not no_videos}
- Language annotation file: {annotation_path}

When the official `h5_agilex_3rgb.json` is supplied, its task steps are converted
to contiguous InfiData segments. Otherwise `subtask=task` is emitted with
`annotation_status=auto_placeholder`.

The script can optionally extract only `trajectory.hdf5` files from RoboMIND
`*.tar.gz.part-*` archives by passing `--extract_root`.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert RoboMIND Agilex HDF5 data to InfiData.")
    parser.add_argument(
        "--robomind_root",
        default="/mnt/workspace/szeluresearch/ELUBrain/RoboMIND",
    )
    parser.add_argument("--out_root", default="/mnt/workspace/InfiData/RoboMIND")
    parser.add_argument(
        "--extract_root",
        default=None,
        help="Extract trajectory.hdf5 from multipart archives here when source is compressed",
    )
    parser.add_argument("--annotation_json", default=None)
    parser.add_argument("--num_episodes", type=int, default=0, help="0 converts all episodes")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--future_seconds", type=float, default=2.0)
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    robomind_root = Path(args.robomind_root).resolve()
    out_root = Path(args.out_root).resolve()
    source_root = robomind_root

    h5_files = discover_hdf5_files(source_root)
    if not h5_files:
        if args.extract_root is None:
            raise FileNotFoundError(
                "No RoboMIND trajectory.hdf5 files found. The provided dataset contains "
                "multipart archives; pass --extract_root to extract them before conversion."
            )
        source_root = Path(args.extract_root).resolve()
        extract_hdf5_archives(robomind_root, source_root)
        h5_files = discover_hdf5_files(source_root)
    if not h5_files:
        raise RuntimeError(f"No RoboMIND HDF5 episodes found under {source_root}")
    if args.num_episodes > 0:
        h5_files = h5_files[: args.num_episodes]

    annotation_search_root = source_root
    annotation_path = find_annotation_json(annotation_search_root, args.annotation_json)
    annotation_lookup = load_annotations(annotation_path)

    all_episode_meta = []
    all_segments = []
    all_stats = []
    skipped = []

    for h5_path in tqdm(h5_files, desc="Converting RoboMIND episodes"):
        try:
            episode_meta, segments, stats = convert_episode(
                h5_path=h5_path,
                annotation_lookup=annotation_lookup,
                output_episode_index=len(all_episode_meta),
                out_root=out_root,
                fps=args.fps,
                no_videos=args.no_videos,
                overwrite=args.overwrite,
                future_seconds=args.future_seconds,
            )
        except Exception as exc:
            if args.strict:
                raise
            skipped.append({"source_file": str(h5_path), "reason": str(exc)})
            continue

        all_episode_meta.append(episode_meta)
        all_segments.extend(segments)
        all_stats.append(stats)

    if not all_episode_meta:
        raise RuntimeError("No valid RoboMIND episodes were converted")

    tasks = {
        str(index): {"task": task, "source_dataset": "RoboMIND"}
        for index, task in enumerate(sorted({item["task"] for item in all_episode_meta}))
    }
    robots = {
        "agilex_cobot_magic": {
            "robot_type": "agilex_cobot_magic",
            "domain": "real",
            "state_dim": int(len(all_stats[0]["state_mean"])),
            "action_dim": int(len(all_stats[0]["action_mean"])),
            "control_mode": "joint",
            "fps": int(args.fps),
            "cameras": [] if args.no_videos else list(CAMERA_MAP),
            "state_description": "puppet left/right joint positions",
            "action_description": "master left/right joint positions",
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
    write_readme(
        out_root,
        num_episodes=len(all_episode_meta),
        fps=args.fps,
        annotation_path=annotation_path,
        no_videos=args.no_videos,
    )

    print(f"[DONE] Converted {len(all_episode_meta)} RoboMIND episodes to {out_root}")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} episodes; see meta/stats.json")


if __name__ == "__main__":
    main()
