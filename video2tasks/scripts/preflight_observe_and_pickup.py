#!/usr/bin/env python3
"""Validate observe_and_pickup inputs without starting model inference."""

import json
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd


DATASET_DIR = Path("/mnt/workspace/InfiData/Rmbench_infi/observe_and_pickup")
EPISODES_PATH = DATASET_DIR / "meta" / "episodes.jsonl"
SEGMENTS_PATH = DATASET_DIR / "meta" / "segments.jsonl"
MEMORY_PATH = DATASET_DIR / "meta" / "memory_segments.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    episodes = load_jsonl(EPISODES_PATH)
    segments = load_jsonl(SEGMENTS_PATH)
    memory = load_jsonl(MEMORY_PATH) if MEMORY_PATH.exists() else []

    errors = []
    checked_videos = 0
    checked_parquets = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        video_rel = (episode.get("video_paths") or {}).get("cam_high")
        parquet_rel = episode.get("parquet_path")

        if not video_rel:
            errors.append(f"episode {episode_index}: missing video_paths.cam_high")
        else:
            video_path = DATASET_DIR / video_rel
            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            if not cap.isOpened() or not ok or frame is None:
                errors.append(f"episode {episode_index}: cannot decode {video_path}")
            else:
                checked_videos += 1
            cap.release()

        if not parquet_rel:
            errors.append(f"episode {episode_index}: missing parquet_path")
        else:
            parquet_path = DATASET_DIR / parquet_rel
            try:
                columns = set(pd.read_parquet(parquet_path).columns)
                required = {"frame_index", "timestamp", "subtask"}
                missing = sorted(required - columns)
                if missing:
                    errors.append(
                        f"episode {episode_index}: parquet missing columns {missing}"
                    )
                else:
                    checked_parquets += 1
            except Exception as exc:
                errors.append(f"episode {episode_index}: cannot read parquet: {exc}")

    subtask_counts = Counter(int(row["episode_index"]) for row in segments)
    memory_counts = Counter(int(row["episode_index"]) for row in memory)

    print(f"episodes: {len(episodes)}")
    print(f"decodable cam_high videos: {checked_videos}")
    print(f"readable parquets: {checked_parquets}")
    print(f"episodes with subtasks: {len(subtask_counts)}")
    print(f"existing memory records: {len(memory)}")
    print(f"episodes with existing memory: {len(memory_counts)}")

    if errors:
        print("\nPreflight failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    if len(subtask_counts) != len(episodes):
        raise SystemExit("Preflight failed: some episodes have no subtask context")

    print("\nPreflight passed.")
    if memory:
        print(
            "Warning: memory annotations already exist. Back up metadata and "
            "parquets before running with write_back=true."
        )


if __name__ == "__main__":
    main()
