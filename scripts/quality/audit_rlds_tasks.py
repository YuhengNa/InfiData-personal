#!/usr/bin/env python3
"""Audit episode-level task membership in an existing TFDS/RLDS dataset.

The script reads serialized ``tf.train.Example`` records without decoding image
features.  It is intended to produce a small manifest that can later drive a
task-disjoint train/seen/unseen repartition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tqdm import tqdm


_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?。！？]+$")


@dataclass(frozen=True)
class EpisodeTask:
    split: str
    episode_index: int
    source_episode_index: int
    task: str
    task_key: str
    tasks: tuple[str, ...]
    task_keys: tuple[str, ...]
    source_dataset_name: str


def _bytes_value(example: tf.train.Example, key: str, default: str = "") -> str:
    values = example.features.feature[key].bytes_list.value
    return bytes(values[0]).decode("utf-8", errors="replace") if values else default


def _int_value(example: tf.train.Example, key: str, default: int = -1) -> int:
    values = example.features.feature[key].int64_list.value
    return int(values[0]) if values else default


def normalize_task(task: str) -> str:
    """Return a task identity key while retaining original text as an alias."""
    task = unicodedata.normalize("NFKC", task)
    task = _WHITESPACE_RE.sub(" ", task).strip().casefold()
    return _TRAILING_SENTENCE_PUNCTUATION_RE.sub("", task).rstrip()


def stable_task_id(task_key: str) -> str:
    return hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:16]


def _tasks_from_example(example: tf.train.Example, primary_task: str) -> tuple[str, ...]:
    raw = _bytes_value(example, "episode_metadata/tasks_json", "")
    tasks: list[str] = []
    if raw:
        try:
            value = json.loads(raw)
            if isinstance(value, list):
                tasks = [str(item).strip() for item in value if str(item).strip()]
        except json.JSONDecodeError:
            pass
    if primary_task and primary_task not in tasks:
        tasks.insert(0, primary_task)
    return tuple(dict.fromkeys(tasks or ([primary_task] if primary_task else [])))


def read_shard(path: Path, split: str) -> list[EpisodeTask]:
    records: list[EpisodeTask] = []
    for raw in tf.compat.v1.io.tf_record_iterator(str(path)):
        example = tf.train.Example.FromString(raw)
        task = _bytes_value(example, "episode_metadata/task").strip()
        task_key = normalize_task(task)
        tasks = _tasks_from_example(example, task)
        records.append(
            EpisodeTask(
                split=split,
                episode_index=_int_value(example, "episode_metadata/episode_index"),
                source_episode_index=_int_value(example, "episode_metadata/source_episode_index"),
                task=task,
                task_key=task_key,
                tasks=tasks,
                task_keys=tuple(dict.fromkeys(normalize_task(item) for item in tasks)),
                source_dataset_name=_bytes_value(example, "episode_metadata/source_dataset_name").strip(),
            )
        )
    return records


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


def load_episode_manifest(path: Path) -> list[EpisodeTask]:
    episodes: list[EpisodeTask] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task = str(row.get("task", "")).strip()
            tasks = tuple(str(item).strip() for item in row.get("tasks", []) if str(item).strip())
            episodes.append(
                EpisodeTask(
                    split=str(row["source_split"]),
                    episode_index=int(row["episode_index"]),
                    source_episode_index=int(row.get("source_episode_index", -1)),
                    task=task,
                    task_key=normalize_task(task),
                    tasks=tasks,
                    task_keys=tuple(dict.fromkeys(normalize_task(item) for item in tasks)),
                    source_dataset_name=str(row.get("source_dataset_name", "")).strip(),
                )
            )
    return episodes


def write_outputs(output_dir: Path, dataset_dir: Path, episodes: list[EpisodeTask], shard_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[EpisodeTask]] = defaultdict(list)
    text_variants: dict[str, Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        by_task[episode.task_key].append(episode)
        text_variants[episode.task_key][episode.task] += 1

    episode_ids: dict[int, list[EpisodeTask]] = defaultdict(list)
    for episode in episodes:
        episode_ids[episode.episode_index].append(episode)
    duplicate_ids = {key: value for key, value in episode_ids.items() if len(value) > 1}

    split_tasks = {
        split: {episode.task_key for episode in episodes if episode.split == split}
        for split in sorted({episode.split for episode in episodes})
    }
    train_tasks = split_tasks.get("train", set())
    old_unseen_tasks = split_tasks.get("unseen_test", set())

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "source_splits": sorted(split_tasks),
        "num_shards_scanned": shard_count,
        "num_episodes": len(episodes),
        "num_unique_episode_indices": len(episode_ids),
        "num_duplicate_episode_indices": len(duplicate_ids),
        "num_primary_tasks": len(by_task),
        "num_distinct_primary_task_texts": len({episode.task for episode in episodes}),
        "num_empty_primary_tasks": sum(not episode.task_key for episode in episodes),
        "num_multi_task_episodes": sum(len(episode.task_keys) > 1 for episode in episodes),
        "episodes_by_source_split": dict(sorted(Counter(item.split for item in episodes).items())),
        "tasks_by_source_split": {key: len(value) for key, value in split_tasks.items()},
        "current_train_unseen_task_overlap": len(train_tasks & old_unseen_tasks),
        "current_train_only_tasks": len(train_tasks - old_unseen_tasks),
        "current_unseen_only_tasks": len(old_unseen_tasks - train_tasks),
        "normalization": "Unicode NFKC, collapse whitespace, strip, casefold, remove trailing sentence punctuation",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "tasks.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "task_id",
            "task_key",
            "representative_task",
            "num_episodes",
            "num_current_train",
            "num_current_unseen_test",
            "num_text_variants",
            "text_variants_json",
            "num_source_datasets",
            "source_dataset_names_json",
            "min_episode_index",
            "max_episode_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        ordered = sorted(by_task.items(), key=lambda item: (-len(item[1]), item[0]))
        for task_key, items in ordered:
            variants = text_variants[task_key]
            representative = variants.most_common(1)[0][0] if variants else ""
            sources = sorted({item.source_dataset_name for item in items if item.source_dataset_name})
            indices = [item.episode_index for item in items]
            split_counts = Counter(item.split for item in items)
            writer.writerow(
                {
                    "task_id": stable_task_id(task_key),
                    "task_key": task_key,
                    "representative_task": representative,
                    "num_episodes": len(items),
                    "num_current_train": split_counts["train"],
                    "num_current_unseen_test": split_counts["unseen_test"],
                    "num_text_variants": len(variants),
                    "text_variants_json": json.dumps(dict(variants.most_common()), ensure_ascii=False),
                    "num_source_datasets": len(sources),
                    "source_dataset_names_json": json.dumps(sources, ensure_ascii=False),
                    "min_episode_index": min(indices),
                    "max_episode_index": max(indices),
                }
            )

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in sorted(episodes, key=lambda item: item.episode_index):
            row = {
                "episode_index": episode.episode_index,
                "source_episode_index": episode.source_episode_index,
                "source_split": episode.split,
                "task_id": stable_task_id(episode.task_key),
                "task": episode.task,
                "task_key": episode.task_key,
                "tasks": list(episode.tasks),
                "task_keys": list(episode.task_keys),
                "source_dataset_name": episode.source_dataset_name,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if duplicate_ids:
        with (output_dir / "duplicate_episode_indices.jsonl").open("w", encoding="utf-8") as handle:
            for episode_index, items in sorted(duplicate_ids.items()):
                handle.write(
                    json.dumps(
                        {
                            "episode_index": episode_index,
                            "occurrences": [
                                {"split": item.split, "task": item.task, "source": item.source_dataset_name}
                                for item in items
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-episodes-manifest",
        type=Path,
        default=None,
        help="Rebuild summaries from a previous episodes.jsonl without rescanning TFRecords.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "unseen_test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=0, help="0 scans every shard; intended for smoke tests.")
    args = parser.parse_args()

    if args.input_episodes_manifest is not None:
        episodes = load_episode_manifest(args.input_episodes_manifest)
        shard_count = len(discover_shards(args.dataset_dir, args.splits))
        write_outputs(args.output_dir, args.dataset_dir, episodes, shard_count)
        print(f"Rebuilt task audit for {len(episodes):,} episodes from {args.input_episodes_manifest}")
        return

    shards = discover_shards(args.dataset_dir, args.splits)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]

    episodes: list[EpisodeTask] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = executor.map(lambda item: read_shard(*item), shards)
        for shard_episodes in tqdm(results, total=len(shards), desc="Auditing RLDS task metadata", unit="shard"):
            episodes.extend(shard_episodes)

    write_outputs(args.output_dir, args.dataset_dir, episodes, len(shards))
    print(f"Audited {len(episodes):,} episodes from {len(shards):,} shards into {args.output_dir}")


if __name__ == "__main__":
    main()
