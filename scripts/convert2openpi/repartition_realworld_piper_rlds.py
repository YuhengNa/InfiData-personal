#!/usr/bin/env python3
"""Repartition piper30 RLDS into task-unseen and trajectory-seen validation.

The source ``train`` and ``unseen_test`` splits together contain every unique
episode.  The old ``seen_test`` is intentionally ignored because it duplicates
episodes in the old train split.  Serialized examples are copied directly, so
JPEG image features are never decoded or re-encoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tqdm import tqdm


_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?。！？]+$")
_OUTPUT_SPLITS = ("train", "seen_test", "unseen_test")


def normalize_task(task: str) -> str:
    task = unicodedata.normalize("NFKC", task)
    task = _WHITESPACE_RE.sub(" ", task).strip().casefold()
    return _TRAILING_SENTENCE_PUNCTUATION_RE.sub("", task).rstrip()


def stable_task_id(task_key: str) -> str:
    return hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:16]


def stable_episode_score(seed: int, task_key: str, episode_index: int) -> bytes:
    value = f"{seed}\0{task_key}\0{episode_index}".encode("utf-8")
    return hashlib.sha256(value).digest()


def sample_count(total: int, ratio: float) -> int:
    if not 0 <= ratio < 1:
        raise ValueError(f"seen ratio must be in [0, 1), got {ratio}")
    if total <= 1 or ratio == 0:
        return 0
    return min(max(int(round(total * ratio)), 1), total - 1)


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    source_split: str
    task: str
    task_key: str
    source_dataset_name: str


def load_episode_manifest(path: Path) -> dict[int, EpisodeInfo]:
    episodes: dict[int, EpisodeInfo] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            if episode_index in episodes:
                raise ValueError(f"Duplicate episode_index={episode_index} at {path}:{line_number}")
            task = str(row.get("task", "")).strip()
            task_key = normalize_task(task)
            if not task_key:
                raise ValueError(f"Empty task for episode_index={episode_index}")
            task_keys = {normalize_task(str(item)) for item in row.get("tasks", []) if str(item).strip()}
            if task_keys and task_keys != {task_key}:
                raise ValueError(
                    f"Multi-task episode is not supported: episode_index={episode_index}, task_keys={task_keys}"
                )
            episodes[episode_index] = EpisodeInfo(
                episode_index=episode_index,
                source_split=str(row["source_split"]),
                task=task,
                task_key=task_key,
                source_dataset_name=str(row.get("source_dataset_name", "")).strip(),
            )
    if not episodes:
        raise ValueError(f"No episodes found in {path}")
    return episodes


def build_assignments(
    episodes: dict[int, EpisodeInfo],
    unseen_tasks: Iterable[str],
    *,
    seen_ratio: float,
    seed: int,
) -> tuple[dict[int, str], dict[str, Any]]:
    by_task: dict[str, list[EpisodeInfo]] = defaultdict(list)
    for episode in episodes.values():
        by_task[episode.task_key].append(episode)

    requested_unseen = {normalize_task(task) for task in unseen_tasks}
    missing = requested_unseen - set(by_task)
    if missing:
        raise ValueError(f"Requested unseen task(s) not found: {sorted(missing)}")

    assignments: dict[int, str] = {}
    task_rows: list[dict[str, Any]] = []
    for task_key, task_episodes in sorted(by_task.items()):
        task_episodes.sort(key=lambda item: item.episode_index)
        if task_key in requested_unseen:
            seen_ids: set[int] = set()
            target = "unseen_test"
        else:
            count = sample_count(len(task_episodes), seen_ratio)
            ranked = sorted(
                task_episodes,
                key=lambda item: stable_episode_score(seed, task_key, item.episode_index),
            )
            seen_ids = {item.episode_index for item in ranked[:count]}
            target = "train"

        counts: Counter[str] = Counter()
        for episode in task_episodes:
            split = "seen_test" if episode.episode_index in seen_ids else target
            assignments[episode.episode_index] = split
            counts[split] += 1
        task_rows.append(
            {
                "task_id": stable_task_id(task_key),
                "task_key": task_key,
                "representative_task": task_episodes[0].task,
                "num_episodes": len(task_episodes),
                "is_unseen_task": task_key in requested_unseen,
                "split_counts": dict(sorted(counts.items())),
                "source_dataset_names": sorted(
                    {item.source_dataset_name for item in task_episodes if item.source_dataset_name}
                ),
            }
        )

    counts = Counter(assignments.values())
    summary = {
        "num_source_episodes": len(episodes),
        "seen_ratio_per_train_task": seen_ratio,
        "seed": seed,
        "unseen_task_keys": sorted(requested_unseen),
        "unseen_task_ids": sorted(stable_task_id(task) for task in requested_unseen),
        "split_counts": {split: counts[split] for split in _OUTPUT_SPLITS},
        "tasks": task_rows,
    }
    return assignments, summary


def discover_shards(dataset_dir: Path, splits: Iterable[str]) -> list[tuple[Path, str]]:
    shards: list[tuple[Path, str]] = []
    for split in splits:
        matches: list[Path] = []
        for attempt in range(3):
            matches = sorted(dataset_dir.glob(f"*-{split}.tfrecord-*"))
            if matches:
                break
            if attempt < 2:
                time.sleep(2)
        if not matches:
            raise FileNotFoundError(f"No TFRecord shards found for split={split!r} in {dataset_dir}")
        shards.extend((path, split) for path in matches)
    return shards


def balanced_lengths(total: int, num_shards: int) -> list[int]:
    num_shards = min(max(1, num_shards), total)
    quotient, remainder = divmod(total, num_shards)
    return [quotient + (index < remainder) for index in range(num_shards)]


class SplitWriter:
    def __init__(self, output_dir: Path, dataset_name: str, split: str, total: int, num_shards: int):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.split = split
        self.target_lengths = balanced_lengths(total, num_shards)
        self.payload_bytes = 0
        self.total_written = 0
        self.shard_index = -1
        self.current_count = 0
        self.writer: tf.io.TFRecordWriter | None = None
        self.temp_paths: list[Path] = []
        self.final_paths: list[Path] = []
        self._open_next()

    def _open_next(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.shard_index += 1
        if self.shard_index >= len(self.target_lengths):
            self.writer = None
            return
        stem = (
            f"{self.dataset_name}-{self.split}.tfrecord-"
            f"{self.shard_index:05d}-of-{len(self.target_lengths):05d}"
        )
        final_path = self.output_dir / stem
        temp_path = self.output_dir / f"{stem}.incomplete"
        self.final_paths.append(final_path)
        self.temp_paths.append(temp_path)
        self.current_count = 0
        self.writer = tf.io.TFRecordWriter(str(temp_path))

    def write(self, raw: bytes) -> None:
        if self.writer is None:
            raise RuntimeError(f"Too many records for split={self.split}")
        self.writer.write(raw)
        self.payload_bytes += len(raw)
        self.total_written += 1
        self.current_count += 1
        if self.current_count == self.target_lengths[self.shard_index]:
            self._open_next()

    def close_and_publish(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.total_written != sum(self.target_lengths):
            raise RuntimeError(
                f"Wrong record count for {self.split}: wrote={self.total_written}, "
                f"expected={sum(self.target_lengths)}"
            )
        for temp_path, final_path in zip(self.temp_paths, self.final_paths, strict=True):
            temp_path.rename(final_path)

    def dataset_info(self) -> dict[str, Any]:
        return {
            "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
            "name": self.split,
            "numBytes": str(self.payload_bytes),
            "shardLengths": [str(value) for value in self.target_lengths],
        }


def _int_value(example: tf.train.Example, key: str) -> int:
    values = example.features.feature[key].int64_list.value
    if not values:
        raise KeyError(f"Missing required int64 feature {key!r}")
    return int(values[0])


def write_plan(output_dir: Path, episodes: dict[int, EpisodeInfo], assignments: dict[int, str], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "split_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for episode_index in sorted(episodes):
            episode = episodes[episode_index]
            handle.write(
                json.dumps(
                    {
                        "episode_index": episode_index,
                        "task_id": stable_task_id(episode.task_key),
                        "task": episode.task,
                        "task_key": episode.task_key,
                        "source_dataset_name": episode.source_dataset_name,
                        "old_split": episode.source_split,
                        "new_split": assignments[episode_index],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def repartition(
    source_dir: Path,
    output_dir: Path,
    episodes: dict[int, EpisodeInfo],
    assignments: dict[int, str],
    summary: dict[str, Any],
    *,
    source_splits: tuple[str, ...],
    shard_counts: dict[str, int],
    output_version: str,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_plan(output_dir, episodes, assignments, summary)

    source_info = json.loads((source_dir / "dataset_info.json").read_text(encoding="utf-8"))
    dataset_name = str(source_info["name"])
    expected_counts = Counter(assignments.values())
    writers = {
        split: SplitWriter(output_dir, dataset_name, split, expected_counts[split], shard_counts[split])
        for split in _OUTPUT_SPLITS
    }

    source_shards = discover_shards(source_dir, source_splits)
    seen_source_ids: set[int] = set()
    progress = tqdm(total=len(episodes), desc="Repartitioning piper30 RLDS", unit="episode")
    try:
        for path, old_split in source_shards:
            for raw in tf.compat.v1.io.tf_record_iterator(str(path)):
                example = tf.train.Example.FromString(raw)
                episode_index = _int_value(example, "episode_metadata/episode_index")
                if episode_index not in assignments:
                    raise KeyError(f"Episode {episode_index} from {old_split}/{path.name} is absent from manifest")
                if episode_index in seen_source_ids:
                    raise ValueError(f"Duplicate source episode_index={episode_index}")
                seen_source_ids.add(episode_index)
                writers[assignments[episode_index]].write(raw)
                progress.update(1)
                if progress.n % 25 == 0:
                    progress.set_postfix({split: writers[split].total_written for split in _OUTPUT_SPLITS})
    finally:
        progress.close()

    missing = set(assignments) - seen_source_ids
    if missing:
        raise RuntimeError(f"Missing {len(missing)} manifest episode(s); first={sorted(missing)[:10]}")
    for writer in writers.values():
        writer.close_and_publish()

    shutil.copy2(source_dir / "features.json", output_dir / "features.json")
    source_info["version"] = output_version
    release_notes = dict(source_info.get("releaseNotes", {}))
    release_notes[output_version] = (
        "piper30 task-disjoint unseen_test and trajectory-disjoint seen_test repartition."
    )
    source_info["releaseNotes"] = release_notes
    source_info["splits"] = [writers[split].dataset_info() for split in _OUTPUT_SPLITS]
    (output_dir / "dataset_info.json").write_text(
        json.dumps(source_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "_SUCCESS").write_text("\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, help="Existing TFDS version directory.")
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New TFDS version directory.")
    parser.add_argument("--output-version", default="1.1.0")
    parser.add_argument("--unseen-task", action="append", required=True, help="Task text; may be repeated.")
    parser.add_argument("--seen-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-splits", nargs="+", default=["train", "unseen_test"])
    parser.add_argument("--train-shards", type=int, default=1024)
    parser.add_argument("--seen-test-shards", type=int, default=256)
    parser.add_argument("--unseen-test-shards", type=int, default=256)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    episodes = load_episode_manifest(args.episode_manifest)
    assignments, summary = build_assignments(
        episodes,
        args.unseen_task,
        seen_ratio=args.seen_ratio,
        seed=args.seed,
    )
    summary.update(
        {
            "source_dir": str(args.source_dir),
            "episode_manifest": str(args.episode_manifest),
            "output_dir": str(args.output_dir),
            "output_version": args.output_version,
            "source_splits": list(args.source_splits),
        }
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.plan_only:
        write_plan(args.output_dir, episodes, assignments, summary)
        print(json.dumps(summary["split_counts"], ensure_ascii=False))
        print(f"Wrote split plan to {args.output_dir}")
        return

    repartition(
        args.source_dir,
        args.output_dir,
        episodes,
        assignments,
        summary,
        source_splits=tuple(args.source_splits),
        shard_counts={
            "train": args.train_shards,
            "seen_test": args.seen_test_shards,
            "unseen_test": args.unseen_test_shards,
        },
        output_version=args.output_version,
    )
    print(f"Built repartitioned piper30 dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
