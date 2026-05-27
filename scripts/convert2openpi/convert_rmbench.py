import argparse
import json
import shutil
from pathlib import Path

import cv2
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from tqdm import tqdm


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMAGE_FEATURE_PREFIX = "observation.images"


class VideoFrameReader:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self._captures: dict[Path, cv2.VideoCapture] = {}

    def close(self):
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()

    def read_rgb(self, rel_or_abs_path: str, frame_index: int) -> np.ndarray:
        video_path = Path(rel_or_abs_path)
        if not video_path.is_absolute():
            video_path = self.dataset_root / video_path
        video_path = video_path.resolve()

        if frame_index < 0:
            raise ValueError(f"Negative frame index {frame_index} for {video_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = self._captures.get(video_path)
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self._captures[video_path] = cap

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_episode_parquets(infidata_root: Path) -> list[Path]:
    episode_meta = read_jsonl(infidata_root / "meta" / "episodes.jsonl")
    if episode_meta:
        paths = []
        for item in sorted(episode_meta, key=lambda x: int(x["episode_index"])):
            parquet_rel = item.get("parquet_path")
            if not parquet_rel:
                raise KeyError(f"Missing parquet_path in episode metadata: {item}")
            paths.append(infidata_root / parquet_rel)
        return paths

    return sorted((infidata_root / "data").glob("*/chunk-*/episode_*.parquet"))


def as_float32_vector(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape {arr.shape}")
    return arr


def require_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def infer_image_shape(first_episode: Path, infidata_root: Path) -> tuple[int, int]:
    df = pd.read_parquet(first_episode)
    if df.empty:
        raise ValueError(f"First episode parquet is empty: {first_episode}")

    reader = VideoFrameReader(infidata_root)
    try:
        row = df.iloc[0]
        path_col = "video.cam_high.path"
        frame_col = "video.cam_high.frame_index"
        if path_col not in row or frame_col not in row:
            raise KeyError(f"Missing {path_col}/{frame_col} in {first_episode}")
        img = reader.read_rgb(require_text(row[path_col], path_col), int(row[frame_col]))
    finally:
        reader.close()

    height, width = img.shape[:2]
    return height, width


def create_lerobot_dataset(
    repo_id: str,
    output_root: Path | None,
    fps: int,
    robot_type: str,
    state_dim: int,
    action_dim: int,
    image_height: int,
    image_width: int,
    use_videos: bool,
    overwrite: bool,
) -> LeRobotDataset:
    root = output_root
    dataset_path = root if root is not None else HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        if not overwrite:
            raise FileExistsError(f"LeRobot dataset already exists: {dataset_path}. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
        "subtask": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "memory": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }

    image_dtype = "video" if use_videos else "image"
    for cam in CAMERA_KEYS:
        features[f"{IMAGE_FEATURE_PREFIX}.{cam}"] = {
            "dtype": image_dtype,
            "shape": (3, image_height, image_width),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
    )


def validate_required_columns(df: pd.DataFrame, path: Path):
    required = {
        "observation.state",
        "action",
        "task",
        "subtask",
    }
    for cam in CAMERA_KEYS:
        required.add(f"video.{cam}.path")
        required.add(f"video.{cam}.frame_index")

    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")


def convert_episode(
    dataset: LeRobotDataset,
    episode_path: Path,
    infidata_root: Path,
    reader: VideoFrameReader,
    state_dim: int,
    action_dim: int,
):
    df = pd.read_parquet(episode_path)
    if df.empty:
        raise ValueError(f"Episode parquet is empty: {episode_path}")
    validate_required_columns(df, episode_path)

    df = df.sort_values("frame_index", kind="stable").reset_index(drop=True)
    expected = np.arange(len(df))
    actual = df["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError(f"Non-contiguous frame_index in {episode_path}")

    for _, row in df.iterrows():
        task = require_text(row["task"], "task")
        subtask = require_text(row["subtask"], "subtask")
        memory = require_text(row["memory"], "memory") if "memory" in df.columns else subtask

        state = as_float32_vector(row["observation.state"], "observation.state")
        action = as_float32_vector(row["action"], "action")
        if state.shape[0] != state_dim:
            raise ValueError(f"state dim mismatch in {episode_path}: expected {state_dim}, got {state.shape[0]}")
        if action.shape[0] != action_dim:
            raise ValueError(f"action dim mismatch in {episode_path}: expected {action_dim}, got {action.shape[0]}")

        frame = {
            "observation.state": state,
            "action": action,
            "task": task,
            "subtask": subtask,
            "memory": memory,
        }

        for cam in CAMERA_KEYS:
            rel_path = require_text(row[f"video.{cam}.path"], f"video.{cam}.path")
            frame_index = int(row[f"video.{cam}.frame_index"])
            frame[f"{IMAGE_FEATURE_PREFIX}.{cam}"] = reader.read_rgb(rel_path, frame_index)

        dataset.add_frame(frame)

    dataset.save_episode()


def write_readme(output_root: Path, repo_id: str, converted: int, fps: int, use_images: bool):
    image_note = (
        "Images are stored as LeRobot `image` features inside parquet files."
        if use_images
        else "Images are stored as LeRobot `video` features."
    )
    text = f"""# RMBench LeRobot Mini

This dataset was converted from an InfiData RMBench dataset for LeRobot/openpi training.

## Contents

- LeRobot repo id: `{repo_id}`
- Converted episodes: {converted}
- FPS: {fps}
- Robot type: robotwin_dual_arm_sim
- Cameras: cam_high, cam_left_wrist, cam_right_wrist
- {image_note}

## Features

- `observation.state`
- `action`
- `observation.images.cam_high`
- `observation.images.cam_left_wrist`
- `observation.images.cam_right_wrist`
- `subtask`
- `memory`

`memory` is initialized from `subtask` so the openpi data path can be exercised before a separate long-horizon memory annotation pass.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infidata_root", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot_type", type=str, default="robotwin_dual_arm_sim")
    parser.add_argument("--use_images", action="store_true", help="store image files instead of encoding LeRobot videos")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="fail on the first invalid episode instead of skipping")
    args = parser.parse_args()

    infidata_root = Path(args.infidata_root).resolve()
    if not infidata_root.exists():
        raise FileNotFoundError(f"InfiData root not found: {infidata_root}")

    episode_paths = find_episode_parquets(infidata_root)
    if not episode_paths:
        raise FileNotFoundError(f"No InfiData episode parquet files found under {infidata_root}")
    for path in episode_paths:
        if not path.exists():
            raise FileNotFoundError(f"Episode parquet listed in metadata does not exist: {path}")

    if args.num_episodes is not None:
        if args.num_episodes <= 0:
            raise ValueError("--num_episodes must be positive")
        episode_paths = episode_paths[: args.num_episodes]

    first_df = pd.read_parquet(episode_paths[0])
    validate_required_columns(first_df, episode_paths[0])
    state_dim = as_float32_vector(first_df.iloc[0]["observation.state"], "observation.state").shape[0]
    action_dim = as_float32_vector(first_df.iloc[0]["action"], "action").shape[0]
    image_height, image_width = infer_image_shape(episode_paths[0], infidata_root)

    output_root = Path(args.output_root).resolve() if args.output_root else None
    dataset = create_lerobot_dataset(
        repo_id=args.repo_id,
        output_root=output_root,
        fps=args.fps,
        robot_type=args.robot_type,
        state_dim=state_dim,
        action_dim=action_dim,
        image_height=image_height,
        image_width=image_width,
        use_videos=not args.use_images,
        overwrite=args.overwrite,
    )

    reader = VideoFrameReader(infidata_root)
    converted = 0
    skipped = []
    try:
        for episode_path in tqdm(episode_paths, desc="Converting InfiData episodes"):
            try:
                convert_episode(
                    dataset=dataset,
                    episode_path=episode_path,
                    infidata_root=infidata_root,
                    reader=reader,
                    state_dim=state_dim,
                    action_dim=action_dim,
                )
            except Exception as exc:
                dataset.clear_episode_buffer()
                if args.strict:
                    raise
                skipped.append({"episode_path": str(episode_path), "reason": str(exc)})
                print(f"[WARN] Skipping {episode_path}: {exc}")
                continue
            converted += 1
    finally:
        reader.close()
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()

    if converted == 0:
        raise RuntimeError("No episodes were converted")

    if skipped:
        skipped_path = dataset.root / "meta" / "infidata_skipped_episodes.json"
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump(skipped, f, ensure_ascii=False, indent=2)

    write_readme(
        output_root=dataset.root,
        repo_id=args.repo_id,
        converted=converted,
        fps=args.fps,
        use_images=args.use_images,
    )

    print("\n[DONE] LeRobot dataset built from InfiData.")
    print(f"Repo id: {args.repo_id}")
    print(f"Output: {dataset.root}")
    print(f"Converted episodes: {converted}")
    if skipped:
        print(f"Skipped episodes: {len(skipped)}")


if __name__ == "__main__":
    main()
