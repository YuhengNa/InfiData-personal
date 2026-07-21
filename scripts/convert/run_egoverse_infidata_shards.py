#!/usr/bin/env python3
"""Run EgoVerse Zarr -> InfiData conversion in fixed, non-overlapping shards.

The important part is the manifest: we first snapshot the sorted input episode
list, then every shard indexes into that immutable list. This avoids accidental
overlap when the source directory changes or when multiple processes start at
slightly different times.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("/mnt/workspace/wudi/ELUBrain/EgoVerseData")
DEFAULT_OUTPUT_BASE = Path("/mnt/workspace/InfiData")
DEFAULT_MANIFEST = Path("/mnt/workspace/InfiData/EgoVerse_full_manifest.jsonl")
# Keep three shards for the current snapshot and clip the final range to the
# manifest length.
DEFAULT_RANGES = "0:20000,20000:40000,40000:70000"
CONVERTER_PATH = Path(__file__).resolve().with_name("convert_egoverse_zarr_to_infidata.py")


def load_converter():
    spec = importlib.util.spec_from_file_location("egoverse_infidata_converter", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load converter from {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_ranges(value: str) -> list[tuple[int, int]]:
    ranges = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        start_s, end_s = item.split(":", 1)
        start = int(start_s)
        end = int(end_s)
        if start < 0 or end <= start:
            raise ValueError(f"Invalid range: {item}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("At least one range is required.")
    return ranges


def build_manifest(input_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, episode_path in enumerate(sorted(p for p in input_root.iterdir() if p.is_dir())):
        records.append(
            {
                "source_episode_index": index,
                "source_episode_id": episode_path.name,
                "path": str(episode_path),
            }
        )
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    records = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_ranges(records: list[dict[str, Any]], ranges: list[tuple[int, int]]) -> None:
    total = len(records)
    seen: dict[str, tuple[int, int]] = {}
    for shard_idx, (start, end) in enumerate(ranges, start=1):
        if start >= total:
            raise ValueError(f"Shard {shard_idx} starts beyond manifest length: {start} >= {total}")
        clipped_end = min(end, total)
        for record in records[start:clipped_end]:
            episode_id = record["source_episode_id"]
            if episode_id in seen:
                prev_shard, prev_index = seen[episode_id]
                raise ValueError(
                    f"Duplicate source_episode_id across shards: {episode_id} "
                    f"in shard {prev_shard} and {shard_idx}; previous index {prev_index}"
                )
            seen[episode_id] = (shard_idx, int(record["source_episode_index"]))


def shard_output_dir(output_base: Path, prefix: str, shard_idx: int) -> Path:
    return output_base / f"{prefix}_{shard_idx}"


def convert_shard(
    shard_idx: int,
    start: int,
    end: int,
    records: list[dict[str, Any]],
    output_base: Path,
    prefix: str,
    clean_output: bool,
    overwrite: bool,
    action_source_horizon: int,
    action_chunk_length: int,
    action_stride: int | None,
) -> None:
    converter = load_converter()
    out_root = shard_output_dir(output_base, prefix, shard_idx)
    if clean_output and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    converted = []
    errors = []
    shard_records = records[start:min(end, len(records))]
    t0 = time.perf_counter()
    for local_i, record in enumerate(shard_records):
        source_episode_index = int(record["source_episode_index"])
        episode_index = start + len(converted)
        episode_path = Path(record["path"])
        try:
            item = converter.convert_episode(
                episode_path=episode_path,
                out_root=out_root,
                episode_index=episode_index,
                overwrite=overwrite,
                action_source_horizon=action_source_horizon,
                action_chunk_length=action_chunk_length,
                action_stride=action_stride,
                source_episode_index=source_episode_index,
            )
            converted.append(item)
            print(
                f"[SHARD {shard_idx}] OK {len(converted)}/{len(shard_records)} "
                f"episode_index={episode_index} source_index={source_episode_index} "
                f"source={record['source_episode_id']}",
                flush=True,
            )
        except Exception as exc:
            error = {
                "episode_path": str(episode_path),
                "episode_index": int(episode_index),
                "source_episode_index": source_episode_index,
                "source_episode_id": record["source_episode_id"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(
                f"[SHARD {shard_idx}] SKIP source_index={source_episode_index} "
                f"source={record['source_episode_id']}: {error['error']}",
                flush=True,
            )

    if converted:
        converter.write_dataset_meta(out_root, converted)
    if errors:
        converter.write_jsonl(out_root / "meta" / "conversion_errors.jsonl", errors)
    elapsed = time.perf_counter() - t0
    print(
        f"[SHARD {shard_idx}] DONE converted={len(converted)} skipped={len(errors)} "
        f"elapsed={elapsed:.1f}s output={out_root}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--prefix", default="EgoVerse_full")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ranges", default=DEFAULT_RANGES)
    parser.add_argument("--reuse-manifest", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--action-source-horizon", type=int, default=30)
    parser.add_argument("--action-chunk-length", type=int, default=100)
    parser.add_argument("--action-stride", type=int, default=None)
    args = parser.parse_args()

    ranges = parse_ranges(args.ranges)
    if args.reuse_manifest:
        records = read_manifest(args.manifest)
        manifest_action = "reused"
    else:
        records = build_manifest(args.input_root, args.manifest)
        manifest_action = "written"

    validate_ranges(records, ranges)
    total = len(records)
    covered = sum(max(0, min(end, total) - start) for start, end in ranges)
    covered_ids = {
        records[i]["source_episode_id"]
        for start, end in ranges
        for i in range(start, min(end, total))
    }

    print(f"manifest_{manifest_action}: {args.manifest}")
    print(f"manifest_records: {total}")
    print(f"requested_ranges: {ranges}")
    print(f"covered_records: {covered}")
    print(f"covered_unique_source_ids: {len(covered_ids)}")
    print(f"remaining_after_max_range: {max(0, total - max(end for _, end in ranges))}")
    for shard_idx, (start, end) in enumerate(ranges, start=1):
        clipped_end = min(end, total)
        first = records[start]["source_episode_id"] if start < total else None
        last = records[clipped_end - 1]["source_episode_id"] if clipped_end > start else None
        print(
            f"shard_{shard_idx}: input[{start}:{end}] clipped_end={clipped_end} "
            f"count={max(0, clipped_end - start)} output={shard_output_dir(args.output_base, args.prefix, shard_idx)} "
            f"first={first} last={last}"
        )

    if args.dry_run:
        return

    processes = []
    for shard_idx, (start, end) in enumerate(ranges, start=1):
        process = mp.Process(
            target=convert_shard,
            args=(
                shard_idx,
                start,
                end,
                records,
                args.output_base,
                args.prefix,
                args.clean_output,
                args.overwrite,
                args.action_source_horizon,
                args.action_chunk_length,
                args.action_stride,
            ),
        )
        process.start()
        processes.append(process)

    failed = False
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed = True
            print(f"[ERROR] shard process pid={process.pid} exitcode={process.exitcode}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
