import argparse
import io
import json
import os
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


IGNORED_DIR_NAMES = {"._____temp"}

ROBOT_TYPE_BY_EMBODIMENT = {
    "h5_agilex_3rgb": "agilex_cobot_magic",
    "h5_franka_1rgb": "franka_panda",
    "h5_franka_3rgb": "franka_panda",
    "h5_franka_fr3_dual": "franka_fr3_dual",
    "h5_simulation": "franka_sim",
    "h5_sim_franka_3rgb": "franka_sim",
    "h5_sim_tienkung_1rgb": "tienkung_humanoid",
    "h5_tienkung_gello_1rgb": "tienkung_humanoid",
    "h5_tienkung_prod1_gello_1rgb": "tienkung_humanoid",
    "h5_tienkung_xsens_1rgb": "tienkung_humanoid",
    "h5_ur_1rgb": "ur5e",
}

ROBOT_TYPE_BY_SCHEMA = {
    "agilex_dual_arm": "agilex_cobot_magic",
    "master_puppet_joint_position": "generic_master_puppet",
    "simulation_franka_joint_position": "franka_sim",
    "tiangong_joint_position": "tienkung_humanoid",
    "master_joint_position": "generic_joint_position",
}

BGR_IMAGE_EMBODIMENTS = {
    "h5_franka_1rgb",
    "h5_franka_3rgb",
    "h5_franka_fr3_dual",
    "h5_ur_1rgb",
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


def is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


def shard_items(items: list, num_shards: int, shard_index: int) -> list:
    if num_shards <= 1:
        return items
    return [item for index, item in enumerate(items) if index % num_shards == shard_index]


def multipart_groups(
    root: Path,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[tuple[str, list[Path], Path]]:
    groups = defaultdict(list)
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = [dirname for dirname in dirnames if dirname not in IGNORED_DIR_NAMES]
        if is_ignored_path(current_path):
            continue
        for filename in filenames:
            if re.fullmatch(r".+\.tar\.gz\.part-[a-z]+", filename) is None:
                continue
            path = current_path / filename
            stem = filename.split(".tar.gz.part-")[0]
            groups[(path.parent, stem)].append(path)

    ready = []
    for (parts_dir, stem), paths in sorted(groups.items(), key=lambda item: str(item[0])):
        paths = sorted(paths, key=lambda path: path.name)
        relative_parent = parts_dir.relative_to(root)
        ready.append((stem, paths, relative_parent))
    return shard_items(ready, num_shards=num_shards, shard_index=shard_index)


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


def is_hdf5_filename(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".hdf5") or lowered.endswith(".h5")


def clean_partial_archive_outputs(target_root: Path, roots: set[str]) -> None:
    for root in roots:
        if not root or root == ".extract_markers":
            continue
        path = target_root / root
        if path.exists():
            shutil.rmtree(path)


def extract_hdf5_archives(
    robomind_root: Path,
    extract_root: Path,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict]:
    groups = multipart_groups(
        robomind_root,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if not groups:
        raise FileNotFoundError(f"No multipart RoboMIND archives found under {robomind_root}")

    skipped_archives = []
    desc = f"Extracting RoboMIND HDF5 archives shard {shard_index}/{num_shards}"
    for stem, paths, relative_parent in tqdm(groups, desc=desc):
        target_root = extraction_target(extract_root, relative_parent)
        marker = target_root / ".extract_markers" / f"{stem}.ok"
        if marker.exists():
            continue

        target_root.mkdir(parents=True, exist_ok=True)
        clean_partial_archive_outputs(target_root, {stem})
        reader = MultiPartReader(paths)
        extracted = 0
        touched_roots = set()
        try:
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not is_hdf5_filename(member.name):
                        continue
                    relative = safe_member_name(member.name)
                    if relative is None:
                        continue
                    if relative.parts:
                        touched_roots.add(relative.parts[0])
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    destination = target_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as output:
                        shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
                    extracted += 1
        except Exception as exc:
            clean_partial_archive_outputs(target_root, touched_roots or {stem})
            skipped_archives.append(
                {
                    "archive_group": stem,
                    "relative_parent": str(relative_parent),
                    "parts": [str(path) for path in paths],
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        finally:
            reader.close()

        if extracted == 0:
            clean_partial_archive_outputs(target_root, touched_roots or {stem})
            skipped_archives.append(
                {
                    "archive_group": stem,
                    "relative_parent": str(relative_parent),
                    "parts": [str(path) for path in paths],
                    "reason": "No .hdf5 or .h5 files found in archive group",
                }
            )
            continue
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{extracted}\n", encoding="utf-8")
    return skipped_archives


def discover_completed_extracted_hdf5_files(
    robomind_root: Path,
    extract_root: Path,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[Path]:
    files = []
    for stem, _paths, relative_parent in multipart_groups(
        robomind_root,
        num_shards=num_shards,
        shard_index=shard_index,
    ):
        target_root = extraction_target(extract_root, relative_parent)
        marker = target_root / ".extract_markers" / f"{stem}.ok"
        if not marker.exists():
            continue
        group_root = target_root / stem
        if not group_root.exists():
            continue
        files.extend(discover_hdf5_files(group_root))
    return sorted(files)


def discover_hdf5_files(root: Path) -> list[Path]:
    hdf5_files = []
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in IGNORED_DIR_NAMES and not dirname.endswith("_compressed")
        ]
        if is_ignored_path(current_path):
            continue
        for filename in filenames:
            if is_hdf5_filename(filename):
                hdf5_files.append(current_path / filename)
    return sorted(hdf5_files)


def find_annotation_jsons(root: Path, explicit: str | None) -> list[Path]:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"RoboMIND annotation JSON not found: {path}")
        return [path]

    candidates = sorted(
        path
        for path in root.rglob("language_description_annotation_json/*.json")
        if not is_ignored_path(path)
    )
    return candidates


def load_annotations(paths: list[Path]) -> dict[str, dict]:
    lookup = {}
    for path in paths:
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            if not isinstance(record, dict) or not record.get("id"):
                continue
            lookup[str(record["id"]).strip("/")] = record.get("response", {})
    return lookup


def embodiment_from_path(path: Path) -> str | None:
    embodiment = None
    for part in path.parts:
        if part.startswith("h5_"):
            embodiment = part
    return embodiment


def annotation_id_from_path(path: Path) -> str | None:
    parts = list(path.parts)
    embodiment_index = None
    for index, part in enumerate(parts):
        if part.startswith("h5_"):
            embodiment_index = index
    if embodiment_index is None:
        return None
    return "/".join(parts[embodiment_index:-1]).strip("/")


def task_from_path(path: Path) -> str:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.startswith("h5_") and index + 1 < len(parts):
            return parts[index + 1].replace("_", " ")
        if part == "failure_data" and index + 1 < len(parts):
            return parts[index + 1].replace("_", " ")
    if "success_episodes" in parts:
        index = parts.index("success_episodes")
        if index > 0:
            return parts[index - 1].replace("_", " ")
    return path.parents[2].name.replace("_", " ")


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


def h5_has(h5_file: h5py.File, name: str) -> bool:
    return name in h5_file


def read_concat(h5_file: h5py.File, names: list[str]) -> np.ndarray:
    missing = [name for name in names if name not in h5_file]
    if missing:
        raise KeyError(f"Missing RoboMIND datasets: {missing}")
    arrays = [h5_file[name][:].astype(np.float32) for name in names]
    length = min(len(array) for array in arrays)
    if length == 0:
        raise ValueError(f"Empty RoboMIND motion datasets: {names}")
    return np.concatenate([array[:length] for array in arrays], axis=1)


def next_state_action(state: np.ndarray) -> np.ndarray:
    if len(state) == 1:
        return state.copy()
    return np.concatenate([state[1:], state[-1:]], axis=0)


def align_to_length(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) == length:
        return array
    if len(array) == 0 or length <= 0:
        raise ValueError(f"Cannot align array of length {len(array)} to {length}")
    indices = np.linspace(0, len(array) - 1, num=length).round().astype(np.int64)
    return array[indices]


def read_motion_data(h5_file: h5py.File) -> tuple[np.ndarray, np.ndarray, dict]:
    if h5_has(h5_file, "puppet/joint_position_left") and h5_has(
        h5_file, "puppet/joint_position_right"
    ):
        state = read_concat(
            h5_file,
            ["puppet/joint_position_left", "puppet/joint_position_right"],
        )
        action_names = ["master/joint_position_left", "master/joint_position_right"]
        action = read_concat(h5_file, action_names) if all(h5_has(h5_file, n) for n in action_names) else next_state_action(state)
        schema = "agilex_dual_arm"
        action_source = "master joint positions" if all(h5_has(h5_file, n) for n in action_names) else "next state"
    elif h5_has(h5_file, "puppet/joint_position"):
        state = h5_file["puppet/joint_position"][:].astype(np.float32)
        if h5_has(h5_file, "master/joint_position"):
            action = h5_file["master/joint_position"][:].astype(np.float32)
            action_source = "master joint positions"
        else:
            action = next_state_action(state)
            action_source = "next state"
        schema = "master_puppet_joint_position"
    elif h5_has(h5_file, "franka/joint_position"):
        state = h5_file["franka/joint_position"][:].astype(np.float32)
        action = next_state_action(state)
        schema = "simulation_franka_joint_position"
        action_source = "next state"
    elif h5_has(h5_file, "tiangong/left_arm_joint_pos_seq") and h5_has(
        h5_file, "tiangong/right_arm_joint_pos_seq"
    ):
        names = [
            "tiangong/left_arm_joint_pos_seq",
            "tiangong/left_hand_joint_pos_seq",
            "tiangong/right_arm_joint_pos_seq",
            "tiangong/right_hand_joint_pos_seq",
        ]
        state = read_concat(h5_file, [name for name in names if h5_has(h5_file, name)])
        action = next_state_action(state)
        schema = "tiangong_joint_position"
        action_source = "next state"
    elif h5_has(h5_file, "master/joint_position"):
        state = h5_file["master/joint_position"][:].astype(np.float32)
        action = next_state_action(state)
        schema = "master_joint_position"
        action_source = "next state"
    else:
        raise KeyError("Unsupported RoboMIND HDF5 schema; no supported joint position data found")

    length = min(len(state), len(action))
    if length == 0:
        raise ValueError("Empty RoboMIND state/action arrays")
    state = state[:length]
    action = action[:length]
    return state, action, {
        "schema": schema,
        "action_source": action_source,
        "state_source": "joint positions",
    }


def discover_rgb_cameras(h5_file: h5py.File) -> dict[str, str]:
    group = h5_file.get("observations/rgb_images")
    if group is None:
        return {}

    cameras = {}
    for source_name in sorted(group.keys()):
        source_path = f"observations/rgb_images/{source_name}"
        if source_path == "observations/rgb_images/camera_front":
            infi_name = "cam_high"
        elif source_path == "observations/rgb_images/camera_left_wrist":
            infi_name = "cam_left_wrist"
        elif source_path == "observations/rgb_images/camera_right_wrist":
            infi_name = "cam_right_wrist"
        else:
            infi_name = re.sub(r"[^0-9A-Za-z_]+", "_", source_name)
            if infi_name.startswith("camera_"):
                infi_name = "cam_" + infi_name[len("camera_") :]
        cameras[infi_name] = source_path
    return cameras


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


def decode_image(value, raw_is_bgr: bool) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.ndim == 3:
        image = value
        if image.shape[2] != 3:
            raise ValueError(f"Unsupported raw RoboMIND image shape: {image.shape}")
        return image if raw_is_bgr else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    raw = np.asarray(value, dtype=np.uint8).reshape(-1)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode RoboMIND image frame")
    return image


def export_camera_video(
    dataset,
    destination: Path,
    fps: int,
    overwrite: bool,
    raw_is_bgr: bool,
    max_frames: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    if destination.exists():
        destination.unlink()

    if max_frames <= 0:
        raise ValueError(f"Cannot export empty video: {destination}")

    first = decode_image(dataset[0], raw_is_bgr=raw_is_bgr)
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
        for index in range(1, max_frames):
            image = decode_image(dataset[index], raw_is_bgr=raw_is_bgr)
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
    finally:
        writer.release()


def episode_domain(embodiment: str | None, schema: str) -> str:
    if embodiment and ("sim" in embodiment or "simulation" in embodiment):
        return "sim"
    if schema.startswith("simulation"):
        return "sim"
    return "real"


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
    embodiment = embodiment_from_path(h5_path)
    raw_is_bgr = embodiment in BGR_IMAGE_EMBODIMENTS

    with h5py.File(h5_path, "r") as h5_file:
        state, action, motion_meta = read_motion_data(h5_file)
        camera_map = discover_rgb_cameras(h5_file)
        if not camera_map and not no_videos:
            raise KeyError(f"No RoboMIND RGB cameras found in {h5_path}")

        if camera_map:
            T = min(len(h5_file[source_camera]) for source_camera in camera_map.values())
        else:
            T = len(state)
        state = align_to_length(state, T)
        action = align_to_length(action, T)
        if T == 0:
            raise ValueError(f"Empty RoboMIND episode: {h5_path}")

        task = ""
        for language_key in ("language_raw", "language_instruction"):
            if language_key in h5_file:
                task = decode_h5_text(h5_file[language_key][()])
                if task:
                    break
        if not task:
            task = str(annotation.get("task_summary", "")).strip()
        if not task:
            task = task_from_path(h5_path)

        segments = make_segments(
            annotation=annotation,
            task=task,
            T=T,
            output_episode_index=output_episode_index,
            source_file=h5_path,
        )

        video_paths = {}
        if not no_videos:
            for infi_camera, source_camera in camera_map.items():
                if len(h5_file[source_camera]) < T:
                    raise ValueError(
                        f"RoboMIND camera length mismatch in {h5_path}: "
                        f"{source_camera} has {len(h5_file[source_camera])}, expected at least {T}"
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
                    raw_is_bgr=raw_is_bgr,
                    max_frames=T,
                )
                video_paths[infi_camera] = str(rel)

    is_success = "success_episodes" in h5_path.parts
    is_failure = (
        "failure_data" in h5_path.parts
        or "failure_episodes" in h5_path.parts
        or "fail_episodes" in h5_path.parts
    )
    success = is_success or not is_failure
    mistake = not success
    quality = 5 if success else 2
    outcome_status = "source_path_label" if is_success or is_failure else "auto_placeholder"
    robot_type = ROBOT_TYPE_BY_EMBODIMENT.get(
        embodiment or "",
        ROBOT_TYPE_BY_SCHEMA.get(motion_meta["schema"], motion_meta["schema"]),
    )
    domain = episode_domain(embodiment, motion_meta["schema"])

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
            "robot_type": robot_type,
            "embodiment": embodiment or motion_meta["schema"],
            "control_mode": "joint",
            "quality": int(quality),
            "speed_bin": int(speed_bin),
            "mistake": bool(mistake),
            "success": bool(success),
            "source_dataset": "RoboMIND",
            "domain": domain,
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

    camera_mapping = camera_map if not no_videos else {}
    episode_meta = {
        "episode_index": int(output_episode_index),
        "source_file": str(h5_path),
        "source_annotation_id": annotation_id,
        "parquet_path": str(parquet_path.relative_to(out_root)),
        "task": task,
        "robot_type": robot_type,
        "embodiment": embodiment,
        "schema": motion_meta["schema"],
        "control_mode": "joint",
        "num_frames": int(T),
        "fps": int(fps),
        "duration_sec": float(T / fps),
        "speed_bin": int(speed_bin),
        "quality": int(quality),
        "success": bool(success),
        "mistake": bool(mistake),
        "source_dataset": "RoboMIND",
        "domain": domain,
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "video_paths": video_paths,
        "camera_mapping": camera_mapping,
        "subtask_annotation_status": annotation_status,
        "outcome_annotation_status": outcome_status,
        "annotation_note": (
            f"State uses {motion_meta['state_source']}; action uses {motion_meta['action_source']}. "
            "Simulation and TienKung-only schemas align motion arrays to RGB frame count when needed. "
            "Subtasks use official RoboMIND language-description steps when available."
        ),
    }
    stats = vector_stats(state, action)
    stats["episode_index"] = int(output_episode_index)
    return episode_meta, segments, stats


def build_robot_metadata(all_episode_meta: list[dict], fps: int, no_videos: bool) -> dict:
    robots = {}
    for episode in all_episode_meta:
        key = episode["robot_type"]
        robot = robots.setdefault(
            key,
            {
                "robot_type": key,
                "domains": set(),
                "embodiments": set(),
                "state_dims": set(),
                "action_dims": set(),
                "control_mode": "joint",
                "fps": int(fps),
                "cameras": set(),
                "state_description": "RoboMIND joint-position state, schema-dependent",
                "action_description": "RoboMIND master joint positions when available, otherwise next state",
            },
        )
        robot["domains"].add(episode["domain"])
        if episode.get("embodiment"):
            robot["embodiments"].add(episode["embodiment"])
        robot["state_dims"].add(int(episode["state_dim"]))
        robot["action_dims"].add(int(episode["action_dim"]))
        if not no_videos:
            robot["cameras"].update(episode.get("video_paths", {}).keys())

    normalized = {}
    for key, robot in robots.items():
        normalized[key] = {
            **robot,
            "domains": sorted(robot["domains"]),
            "embodiments": sorted(robot["embodiments"]),
            "state_dims": sorted(robot["state_dims"]),
            "action_dims": sorted(robot["action_dims"]),
            "cameras": sorted(robot["cameras"]),
        }
    return normalized


def write_readme(
    out_root: Path,
    num_episodes: int,
    fps: int,
    annotation_paths: list[Path],
    no_videos: bool,
    num_shards: int,
    shard_index: int,
) -> None:
    annotation_text = "\n".join(f"- {path}" for path in annotation_paths) or "- none"
    text = f"""# RoboMIND InfiData

Converted from RoboMIND HDF5 trajectories.

- Episodes: {num_episodes}
- Source dataset: RoboMIND
- Shard: {shard_index} / {num_shards}
- FPS: {fps}
- Videos included: {not no_videos}
- Language annotation files:
{annotation_text}

This conversion supports the full multi-embodiment RoboMIND layout, including
AgileX, Franka, Franka FR3 dual-arm, UR, TienKung, and simulation HDF5 schemas.
When official RoboMIND language-description annotations are available, task
steps are converted to contiguous InfiData segments. Otherwise `subtask=task`
is emitted with `annotation_status=auto_placeholder`.

If `--extract_root` was passed, the converter extracted only `trajectory.hdf5`
files from the sharded `*.tar.gz.part-*` archive groups before conversion.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert full RoboMIND HDF5 data to InfiData.")
    parser.add_argument(
        "--robomind_root",
        default="/mnt/workspace/wudi/ELUBrain/RoboMIND",
    )
    parser.add_argument("--out_root", default="/mnt/workspace/InfiData/RoboMIND")
    parser.add_argument(
        "--extract_root",
        default=None,
        help="Extract *.hdf5/*.h5 files from multipart archives here before conversion",
    )
    parser.add_argument("--annotation_json", default=None)
    parser.add_argument("--num_episodes", type=int, default=0, help="0 converts all episodes")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--future_seconds", type=float, default=2.0)
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    robomind_root = Path(args.robomind_root).resolve()
    out_root = Path(args.out_root).resolve()

    direct_h5_files = shard_items(
        discover_hdf5_files(robomind_root),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    extracted_h5_files = []
    skipped_archives = []
    if args.extract_root is not None:
        extract_root = Path(args.extract_root).resolve()
        skipped_archives = extract_hdf5_archives(
            robomind_root,
            extract_root,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        extracted_h5_files = discover_completed_extracted_hdf5_files(
            robomind_root,
            extract_root,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )

    h5_files = sorted(direct_h5_files + extracted_h5_files)
    if not h5_files:
        raise RuntimeError(f"No RoboMIND HDF5 episodes found under {robomind_root}")
    if args.num_episodes > 0:
        h5_files = h5_files[: args.num_episodes]

    annotation_paths = find_annotation_jsons(robomind_root, args.annotation_json)
    annotation_lookup = load_annotations(annotation_paths)

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
    robots = build_robot_metadata(all_episode_meta, fps=args.fps, no_videos=args.no_videos)
    dataset_stats = {
        "num_episodes": len(all_episode_meta),
        "num_frames": int(sum(item["num_frames"] for item in all_episode_meta)),
        "num_skipped_episodes": len(skipped),
        "num_skipped_archives": len(skipped_archives),
        "num_input_hdf5_files": len(h5_files),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "episodes": all_stats,
        "skipped_episodes": skipped,
        "skipped_archives": skipped_archives,
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
        annotation_paths=annotation_paths,
        no_videos=args.no_videos,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    print(f"[DONE] Converted {len(all_episode_meta)} RoboMIND episodes to {out_root}")
    print(f"[INFO] Input HDF5 files in this shard: {len(h5_files)}")
    if skipped_archives:
        print(f"[INFO] Skipped {len(skipped_archives)} archive groups; see meta/stats.json")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} episodes; see meta/stats.json")


if __name__ == "__main__":
    main()
