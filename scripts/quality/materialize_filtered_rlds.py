#!/usr/bin/env python3
"""Create an independent RLDS copy using final episode decisions.

Train TFRecord shards are rewritten without decoding or re-encoding examples.
Other splits and metadata files are copied byte-for-byte. Source files are only
opened for reading. Per-shard checkpoints make the operation resumable.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import mmap
import os
import shutil
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm


TRAIN_TOKEN = "-train.tfrecord-"
GLOBAL_KEY = b"episode_metadata/global_episode_key"
EPISODE_INDEX = b"episode_metadata/episode_index"
SOURCE_EPISODE_INDEX = b"episode_metadata/source_episode_index"


def _read_varint(data: memoryview, position: int, end: int) -> tuple[int, int]:
    value = shift = 0
    while position < end and shift < 70:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7
    raise ValueError("Invalid protobuf varint")


def _fields(
    data: memoryview, start: int, end: int
) -> Iterator[tuple[int, int, int, int | None]]:
    position = start
    while position < end:
        tag, position = _read_varint(data, position, end)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError("Invalid protobuf field number 0")
        if wire == 0:
            value, position = _read_varint(data, position, end)
            yield number, wire, value, None
        elif wire == 1:
            stop = position + 8
            if stop > end:
                raise ValueError("Truncated protobuf fixed64")
            yield number, wire, position, stop
            position = stop
        elif wire == 2:
            length, position = _read_varint(data, position, end)
            stop = position + length
            if stop > end:
                raise ValueError("Truncated protobuf bytes field")
            yield number, wire, position, stop
            position = stop
        elif wire == 5:
            stop = position + 4
            if stop > end:
                raise ValueError("Truncated protobuf fixed32")
            yield number, wire, position, stop
            position = stop
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")


def _first_bytes(data: memoryview, feature_start: int, feature_end: int) -> bytes | None:
    for number, wire, start, end in _fields(data, feature_start, feature_end):
        if number == 1 and wire == 2 and end is not None:  # Feature.bytes_list
            for child_number, child_wire, value_start, value_end in _fields(data, start, end):
                if child_number == 1 and child_wire == 2 and value_end is not None:
                    return bytes(data[value_start:value_end])
    return None


def _first_int(data: memoryview, feature_start: int, feature_end: int) -> int | None:
    for number, wire, start, end in _fields(data, feature_start, feature_end):
        if number != 3 or wire != 2 or end is None:  # Feature.int64_list
            continue
        for child_number, child_wire, value_start, value_end in _fields(data, start, end):
            if child_number != 1:
                continue
            if child_wire == 0:
                return value_start
            if child_wire == 2 and value_end is not None:
                value, _ = _read_varint(data, value_start, value_end)
                return value
    return None


def extract_episode_identifiers(
    data: memoryview, start: int, end: int, ordinal: int
) -> dict[str, str]:
    """Read episode identifiers without decoding any other Example features."""
    features_span: tuple[int, int] | None = None
    for number, wire, value_start, value_end in _fields(data, start, end):
        if number == 1 and wire == 2 and value_end is not None:
            features_span = (value_start, value_end)
            break
    if features_span is None:
        raise KeyError("Serialized Example has no features")

    global_key: bytes | None = None
    episode_index: int | None = None
    source_episode_index: int | None = None
    for number, wire, entry_start, entry_end in _fields(data, *features_span):
        if number != 1 or wire != 2 or entry_end is None:
            continue
        key: bytes | None = None
        value_span: tuple[int, int] | None = None
        for child_number, child_wire, value_start, value_end in _fields(
            data, entry_start, entry_end
        ):
            if child_number == 1 and child_wire == 2 and value_end is not None:
                key = bytes(data[value_start:value_end])
            elif child_number == 2 and child_wire == 2 and value_end is not None:
                value_span = (value_start, value_end)
        if key == GLOBAL_KEY and value_span:
            global_key = _first_bytes(data, *value_span)
        elif key == EPISODE_INDEX and value_span:
            episode_index = _first_int(data, *value_span)
        elif key == SOURCE_EPISODE_INDEX and value_span:
            source_episode_index = _first_int(data, *value_span)

    identifiers = {"ordinal": str(ordinal)}
    if global_key:
        identifiers["global_episode_key"] = global_key.decode("utf-8")
    if episode_index is not None:
        identifiers["episode_index"] = str(episode_index)
    if source_episode_index is not None:
        identifiers["source_episode_index"] = str(source_episode_index)
    return identifiers


def extract_episode_keys(data: memoryview, start: int, end: int, ordinal: int) -> list[str]:
    identifiers = extract_episode_identifiers(data, start, end, ordinal)
    values: list[str] = []
    for field in ("global_episode_key", "episode_index", "source_episode_index", "ordinal"):
        if field in identifiers and identifiers[field] not in values:
            values.append(identifiers[field])
    return values


def extract_episode_key(data: memoryview, start: int, end: int, ordinal: int) -> str:
    """Return the preferred key, retained as a convenience for diagnostics/tests."""
    return extract_episode_keys(data, start, end, ordinal)[0]


def iter_tfrecord(data: memoryview) -> Iterator[tuple[int, int, int, int]]:
    """Yield (record_start, payload_start, payload_end, record_end)."""
    position, size = 0, len(data)
    while position < size:
        if position + 12 > size:
            raise ValueError(f"Truncated TFRecord header at byte {position}")
        length = struct.unpack_from("<Q", data, position)[0]
        payload_start = position + 12
        payload_end = payload_start + length
        record_end = payload_end + 4
        if record_end > size:
            raise ValueError(f"Truncated TFRecord payload at byte {position}")
        yield position, payload_start, payload_end, record_end
        position = record_end


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            key = str(row["episode"])
            if key in decisions:
                raise ValueError(f"Duplicate decision key {key!r} at {path}:{line_number}")
            decisions[key] = row
    return decisions


def _filter_shard(
    source: Path,
    destination: Path,
    checkpoint: Path,
    decisions: dict[str, dict[str, Any]],
    decision_sha256: str,
    episode_key_field: str,
    ordinal_start: int,
    progress: tqdm,
) -> dict[str, Any]:
    source_stat = source.stat()
    if destination.exists() or checkpoint.exists():
        if not (destination.exists() and checkpoint.exists()):
            raise RuntimeError(f"Incomplete resume state for {destination}")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        valid = (
            state["source_size"] == source_stat.st_size
            and state["source_mtime_ns"] == source_stat.st_mtime_ns
            and state["decision_sha256"] == decision_sha256
            and state["output_size"] == destination.stat().st_size
        )
        if not valid:
            raise RuntimeError(f"Stale checkpoint for {destination}")
        progress.update(source_stat.st_size)
        return state

    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    keys: list[str] = []
    kept = deleted = payload_bytes = source_records = 0
    with source.open("rb") as source_handle, temporary.open("wb") as output_handle:
        with mmap.mmap(source_handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            view = memoryview(mapped)
            try:
                for record_start, payload_start, payload_end, record_end in iter_tfrecord(view):
                    identifiers = extract_episode_identifiers(
                        view, payload_start, payload_end, ordinal_start + source_records
                    )
                    key = identifiers.get(episode_key_field)
                    if key is None or key not in decisions:
                        raise KeyError(
                            f"No decision for {episode_key_field}={key!r}; identifiers="
                            f"{identifiers!r} in {source}"
                        )
                    if key in keys:
                        raise ValueError(f"Duplicate episode key {key!r} in {source}")
                    keys.append(key)
                    if decisions[key]["delete"]:
                        deleted += 1
                    else:
                        output_handle.write(view[record_start:record_end])
                        kept += 1
                        payload_bytes += payload_end - payload_start
                    source_records += 1
                    progress.update(record_end - record_start)
            finally:
                view.release()
    temporary.replace(destination)
    state = {
        "source": str(source),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "decision_sha256": decision_sha256,
        "source_records": source_records,
        "kept": kept,
        "deleted": deleted,
        "payload_bytes": payload_bytes,
        "output_size": destination.stat().st_size,
        "keys": keys,
    }
    _write_json(checkpoint, state)
    return state


def _copy_file(source: Path, destination: Path, progress: tqdm) -> None:
    size = source.stat().st_size
    if destination.exists():
        if destination.stat().st_size != size:
            raise RuntimeError(f"Existing destination differs from source: {destination}")
        progress.update(size)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    with source.open("rb") as src, temporary.open("wb") as dst:
        while chunk := src.read(64 * 1024 * 1024):
            dst.write(chunk)
            progress.update(len(chunk))
    shutil.copystat(source, temporary)
    temporary.replace(destination)


def _source_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*") if path.is_file())


def _existing_ancestor(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def _select_rows(rows: list[dict[str, Any]], patterns: list[str]) -> list[dict[str, Any]]:
    if not patterns:
        return rows
    selected = [
        row for row in rows
        if any(fnmatch.fnmatch(row["source_relative_dir"], pattern) for pattern in patterns)
    ]
    if not selected:
        raise ValueError(f"No datasets matched --dataset {patterns}")
    return selected


def _preflight(
    rows: list[dict[str, Any]], source_root: Path, output_root: Path
) -> dict[str, Any]:
    source_resolved, output_resolved = source_root.resolve(), output_root.resolve()
    if source_resolved == output_resolved:
        raise ValueError("Source and output roots must differ")
    for row in rows:
        dataset_dir = (source_root / row["source_relative_dir"]).resolve()
        if (
            output_resolved == dataset_dir
            or output_resolved in dataset_dir.parents
            or dataset_dir in output_resolved.parents
        ):
            raise ValueError(
                f"Output root overlaps source dataset directory {dataset_dir}"
            )
    source_bytes = 0
    for row in rows:
        info_path = source_root / row["source_relative_dir"] / "dataset_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        source_bytes += sum(int(split["numBytes"]) for split in info["splits"])
    required = int(source_bytes * 1.05)
    free = shutil.disk_usage(_existing_ancestor(output_root)).free
    return {
        "datasets": len(rows),
        "episodes": sum(row["episodes"] for row in rows),
        "delete": sum(row["delete"] for row in rows),
        "keep": sum(row["keep"] for row in rows),
        "source_payload_bytes": source_bytes,
        "conservative_required_bytes": required,
        "available_bytes": free,
        "space_ok": free >= required,
        "source_root": str(source_root),
        "output_root": str(output_root),
    }


def _materialize_dataset(
    row: dict[str, Any],
    source_root: Path,
    output_root: Path,
    decisions_root: Path,
    progress: tqdm,
) -> dict[str, Any]:
    relative = Path(row["source_relative_dir"])
    source_dir, output_dir = source_root / relative, output_root / relative
    decisions_path = decisions_root / row["decisions_file"]
    decision_sha256 = _sha256(decisions_path)
    decisions = _load_decisions(decisions_path)
    if len(decisions) != row["episodes"]:
        raise RuntimeError(f"Decision count mismatch for {relative}")

    plan = {
        "policy_version": "P0-P5_20260807",
        "source_dir": str(source_dir.resolve()),
        "source_relative_dir": str(relative),
        "decisions_file": str(decisions_path),
        "decision_sha256": decision_sha256,
        "expected_episodes": row["episodes"],
        "expected_delete": row["delete"],
        "expected_keep": row["keep"],
        "episode_key_field": row["episode_key_field"],
        "source_modified": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "_QUALITY_FILTER_PLAN.json"
    success_path = output_dir / "_QUALITY_FILTER_SUCCESS.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError(f"Output belongs to another filtering plan: {output_dir}")
    if not plan_path.exists():
        _write_json(plan_path, plan)
    if success_path.exists():
        result = json.loads(success_path.read_text(encoding="utf-8"))
        if result["decision_sha256"] != decision_sha256:
            raise RuntimeError(f"Completed output has a different decision manifest: {output_dir}")
        progress.update(sum(path.stat().st_size for path in _source_files(source_dir)))
        return result

    source_info = json.loads((source_dir / "dataset_info.json").read_text(encoding="utf-8"))
    train_info = next(split for split in source_info["splits"] if split["name"] == "train")
    expected_shard_lengths = [int(value) for value in train_info["shardLengths"]]
    train_shards = sorted(source_dir.glob(f"*{TRAIN_TOKEN}*"))
    if len(train_shards) != len(expected_shard_lengths):
        raise RuntimeError(f"Train shard count mismatch for {source_dir}")
    if sum(expected_shard_lengths) != len(decisions):
        raise RuntimeError(f"TFDS metadata/decision count mismatch for {source_dir}")

    checkpoint_dir = output_dir / ".quality_filter_state"
    shard_states: list[dict[str, Any]] = []
    ordinal = 0
    for source_shard, expected_length in zip(train_shards, expected_shard_lengths, strict=True):
        progress.set_postfix_str(str(relative), refresh=False)
        state = _filter_shard(
            source_shard,
            output_dir / source_shard.name,
            checkpoint_dir / f"{source_shard.name}.json",
            decisions,
            decision_sha256,
            row["episode_key_field"],
            ordinal,
            progress,
        )
        if state["source_records"] != expected_length:
            raise RuntimeError(f"Shard length mismatch for {source_shard}")
        shard_states.append(state)
        ordinal += expected_length

    for source_file in _source_files(source_dir):
        if source_file.name == "dataset_info.json" or TRAIN_TOKEN in source_file.name:
            continue
        if source_file.name in {"_SUCCESS", "_QUALITY_FILTER_PLAN.json", "_QUALITY_FILTER_SUCCESS.json"}:
            progress.update(source_file.stat().st_size)
            continue
        _copy_file(source_file, output_dir / source_file.relative_to(source_dir), progress)
    progress.update((source_dir / "dataset_info.json").stat().st_size)

    seen: set[str] = set()
    for state in shard_states:
        overlap = seen.intersection(state["keys"])
        if overlap:
            raise RuntimeError(f"Duplicate source episode key(s): {sorted(overlap)[:5]}")
        seen.update(state["keys"])
    missing = set(decisions) - seen
    if missing:
        raise RuntimeError(f"Missing source episode key(s): {sorted(missing)[:5]}")

    counts = Counter()
    for state in shard_states:
        counts.update({key: state[key] for key in ("source_records", "kept", "deleted", "payload_bytes")})
    if counts["deleted"] != row["delete"] or counts["kept"] != row["keep"]:
        raise RuntimeError(f"Final counts do not match decisions for {relative}: {counts}")

    train_available = counts["kept"] > 0
    if train_available:
        train_info["shardLengths"] = [str(state["kept"]) for state in shard_states]
        train_info["numBytes"] = str(counts["payload_bytes"])
    else:
        source_info["splits"] = [
            split for split in source_info["splits"] if split["name"] != "train"
        ]
    version = str(source_info.get("version", relative.name))
    note = "State/action quality-filtered train split using confirmed P0-P5_20260807 policy."
    release_notes = dict(source_info.get("releaseNotes", {}))
    release_notes[version] = (release_notes.get(version, "").rstrip() + " " + note).strip()
    source_info["releaseNotes"] = release_notes
    _write_json(output_dir / "dataset_info.json", source_info)
    result = {
        **plan,
        "source_records": counts["source_records"],
        "deleted": counts["deleted"],
        "kept": counts["kept"],
        "output_train_payload_bytes": counts["payload_bytes"],
        "train_shards": len(shard_states),
        "train_available": train_available,
    }
    _write_json(success_path, result)
    (output_dir / "_SUCCESS").write_text("\n", encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    rows = _select_rows(rows, args.dataset)
    decisions_root = args.manifest.parent
    preflight = _preflight(rows, args.source_root, args.output_root)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if not preflight["space_ok"]:
        raise RuntimeError("Insufficient output space; source data was not modified")
    if not args.execute:
        print("Preflight only. Add --execute to create the filtered RLDS copy.")
        return

    source_files = {
        row["source_relative_dir"]: _source_files(args.source_root / row["source_relative_dir"])
        for row in rows
    }
    total_bytes = sum(path.stat().st_size for files in source_files.values() for path in files)
    results = []
    with tqdm(total=total_bytes, desc="Materializing filtered RLDS", unit="B", unit_scale=True) as progress:
        for row in rows:
            results.append(
                _materialize_dataset(
                    row, args.source_root, args.output_root, decisions_root, progress
                )
            )
    summary = {
        "policy_version": "P0-P5_20260807",
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "source_modified": False,
        "datasets": len(results),
        "episodes": sum(result["source_records"] for result in results),
        "deleted": sum(result["deleted"] for result in results),
        "kept": sum(result["kept"] for result in results),
        "excluded_from_training": [
            result["source_relative_dir"] for result in results if not result["train_available"]
        ],
        "results": results,
    }
    _write_json(args.output_root / "_QUALITY_FILTER_BATCH_SUMMARY.json", summary)
    _write_json(
        args.output_root / "_TRAIN_DATASETS.json",
        [result["source_relative_dir"] for result in results if result["train_available"]],
    )
    print(json.dumps({key: summary[key] for key in ("datasets", "episodes", "deleted", "kept")}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/data/wudi/InfiData-personal/quality_runs/final_selection/all_rlds/datasets.jsonl"
        ),
    )
    parser.add_argument("--source-root", type=Path, default=Path("/data/wudi/RLDS"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/wudi/RLDS_State_Action_Filtered_20260807"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional source-relative glob; repeat to select multiple configurations.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write output. Without this flag, only validate the plan and disk space.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
