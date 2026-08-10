#!/usr/bin/env python3
"""Build auditable episode decisions from the confirmed P0-P5 policies."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from select_robomind_quality_candidates import apply_policy as apply_robomind_policy


EXPECTED_EPISODES = 355_654
EXPECTED_DELETE = 31_585
P0_DATASETS = {"AgiBot", "realworld_piper_2", "realworld_piper_task_split"}
_PREFIX = re.compile(
    r'^\{"episode":\s*("(?:\\.|[^"\\])*")\s*,\s*'
    r'"num_frames":\s*(\d+)\s*,\s*"failed_rules":\s*(\[[^]]*\])'
)


def _s3_flagged(episode: dict[str, Any]) -> bool:
    return bool(episode["s3_state"]["flagged"] or episode["s3_action"]["flagged"])


def apply_policy(dataset: str, episode: dict[str, Any]) -> dict[str, Any]:
    s2 = bool(episode["s2"]["flagged"])
    s3 = _s3_flagged(episode)
    if dataset in P0_DATASETS:
        code = "P0"
        s1 = bool(episode["s1_state"]["flagged"] or episode["s1_action"]["flagged"])
        flags = {"S1": s1, "S2": s2, "S3": s3}
    elif dataset == "DROID":
        code = "P1"
        flags = {"S1": bool(episode["s1_state"]["flagged"]), "S2": s2, "S3": s3}
    elif dataset in {"EgoVerse_full", "RoboCOIN"}:
        code = "P2"
        flags = {"S1": False, "S2": s2, "S3": s3}
    elif dataset == "RoboMIND_full":
        result = apply_robomind_policy(episode)
        code = {"master_puppet": "P3", "franka_sim": "P4", "tienkung_s38": "P5"}[
            result["group"]
        ]
        return {
            "policy": code,
            "delete": result["delete"],
            "rules": result["rules"],
            "details": {
                key: value
                for key, value in result.items()
                if key not in {"delete", "rules", "group"}
            },
        }
    else:
        raise ValueError(f"No final policy for dataset {dataset!r}")

    rules = [name for name, flagged in flags.items() if flagged]
    return {"policy": code, "delete": bool(rules), "rules": rules, "details": {}}


def _prefix_fields(line: str) -> tuple[str, int, set[str]]:
    match = _PREFIX.match(line)
    if match is None:
        raise ValueError("Unexpected episodes.jsonl field order")
    key = str(json.loads(match.group(1)))
    num_frames = int(match.group(2))
    raw_rules = set(json.loads(match.group(3)))
    return key, num_frames, raw_rules


def _fast_record(dataset: str, line: str) -> tuple[str, int, dict[str, Any]]:
    """Avoid decoding unused, very large S1 hit arrays."""
    key, num_frames, raw_rules = _prefix_fields(line)
    if dataset in P0_DATASETS:
        code, rules = "P0", [rule for rule in ("S1", "S2", "S3") if rule in raw_rules]
    elif dataset == "DROID":
        code = "P1"
        state_s1 = '"s1_state": {"flagged": true' in line
        rules = [rule for rule, flagged in (
            ("S1", state_s1), ("S2", "S2" in raw_rules), ("S3", "S3" in raw_rules)
        ) if flagged]
    elif dataset in {"EgoVerse_full", "RoboCOIN"}:
        code, rules = "P2", [rule for rule in ("S2", "S3") if rule in raw_rules]
    else:
        raise ValueError(f"Fast policy is not available for {dataset!r}")
    return key, num_frames, {
        "policy": code, "delete": bool(rules), "rules": rules, "details": {}
    }


def _source_relative_path(episodes_path: Path, runs_root: Path) -> Path:
    relative = episodes_path.relative_to(runs_root)
    if relative.parts[-2:] != ("train", "episodes.jsonl"):
        raise ValueError(f"Unexpected run layout: {episodes_path}")
    return Path(*relative.parts[:-2])


def build(args: argparse.Namespace) -> dict[str, Any]:
    result_files = sorted(args.runs_root.glob("*/**/train/episodes.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No train/episodes.jsonl under {args.runs_root}")

    expected_total = sum(
        json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))["episodes_scanned"]
        for path in result_files
    )
    datasets: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    robomind: dict[tuple[str, str], dict[str, Any]] = {}
    with args.robomind_decisions.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lookup = (str(row["dataset"]), str(row["episode"]))
            if lookup in robomind:
                raise ValueError(f"Duplicate RoboMIND final decision {lookup}")
            robomind[lookup] = row
    robomind_seen: set[tuple[str, str]] = set()
    progress = tqdm(total=expected_total, desc="Applying final P0-P5 policies", unit="episode")

    for source in result_files:
        source_relative = _source_relative_path(source, args.runs_root)
        dataset = source_relative.parts[0]
        destination = args.output_root / source_relative / "train"
        destination.mkdir(parents=True, exist_ok=True)
        decisions_tmp = destination / "episode_decisions.jsonl.incomplete"
        candidates_tmp = destination / "delete_candidates.jsonl.incomplete"
        counts: Counter[str] = Counter()
        keys: set[str] = set()
        policy_codes: set[str] = set()
        episode_key_fields: set[str] = set()

        with source.open(encoding="utf-8") as handle, \
                decisions_tmp.open("w", encoding="utf-8") as decisions, \
                candidates_tmp.open("w", encoding="utf-8") as candidates:
            for line_number, line in enumerate(handle, start=1):
                if dataset == "RoboMIND_full":
                    key, num_frames, _ = _prefix_fields(line)
                    lookup = (source_relative.parts[1], key)
                    prior = robomind.get(lookup)
                    if prior is None:
                        raise KeyError(f"Missing confirmed RoboMIND decision {lookup}")
                    robomind_seen.add(lookup)
                    result = {
                        "policy": {
                            "master_puppet": "P3",
                            "franka_sim": "P4",
                            "tienkung_s38": "P5",
                        }[prior["group"]],
                        "delete": prior["delete"],
                        "rules": prior["rules"],
                        "details": prior.get("details", {}),
                    }
                    parquet_path = prior.get("parquet_path")
                else:
                    key, num_frames, result = _fast_record(dataset, line)
                    parquet_path = None
                if key in keys:
                    raise ValueError(f"Duplicate episode key {key!r} at {source}:{line_number}")
                keys.add(key)
                episode_key_fields.add(
                    "source_episode_index" if dataset == "DROID" else
                    "global_episode_key" if dataset == "RoboMIND_full" and ":" in key else
                    "episode_index"
                )
                record = {
                    "episode": key,
                    "delete": result["delete"],
                    "rules": result["rules"],
                    "policy": result["policy"],
                    "num_frames": num_frames,
                    "parquet_path": parquet_path,
                    "details": result["details"],
                }
                text = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                decisions.write(text)
                if record["delete"]:
                    candidates.write(text)
                policy_codes.add(record["policy"])
                counts["episodes"] += 1
                counts["delete"] += record["delete"]
                counts["keep"] += not record["delete"]
                for rule in record["rules"]:
                    counts[rule] += 1
                progress.update()

        decisions_path = destination / "episode_decisions.jsonl"
        candidates_path = destination / "delete_candidates.jsonl"
        decisions_tmp.replace(decisions_path)
        candidates_tmp.replace(candidates_path)
        source_summary = json.loads((source.parent / "summary.json").read_text(encoding="utf-8"))
        if counts["episodes"] != source_summary["episodes_scanned"]:
            raise RuntimeError(f"Episode count mismatch for {source_relative}")
        if len(episode_key_fields) != 1:
            raise RuntimeError(
                f"Mixed episode key conventions for {source_relative}: {episode_key_fields}"
            )

        row = {
            "dataset": dataset,
            "source_relative_dir": str(source_relative),
            "split": "train",
            "episode_key_field": next(iter(episode_key_fields)),
            "policy_codes": sorted(policy_codes),
            "decisions_file": str(decisions_path.relative_to(args.output_root)),
            **counts,
            "delete_rate": counts["delete"] / counts["episodes"],
        }
        datasets.append(row)
        totals.update(counts)

    progress.close()
    if robomind_seen != set(robomind):
        missing = set(robomind) - robomind_seen
        raise RuntimeError(f"Unused confirmed RoboMIND decisions: {list(missing)[:5]}")
    if len(datasets) != args.expect_datasets:
        raise RuntimeError(f"Expected {args.expect_datasets} configurations, got {len(datasets)}")
    if totals["episodes"] != args.expect_episodes or totals["delete"] != args.expect_delete:
        raise RuntimeError(
            "Final-policy totals changed: "
            f"episodes={totals['episodes']:,}, delete={totals['delete']:,}; "
            f"expected {args.expect_episodes:,}/{args.expect_delete:,}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_tmp = args.output_root / "datasets.jsonl.incomplete"
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        for row in datasets:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest_tmp.replace(args.output_root / "datasets.jsonl")
    summary = {
        "policy_version": "P0-P5_20260807",
        "source_data_modified": False,
        "configurations": len(datasets),
        "total": dict(totals),
        "delete_rate": totals["delete"] / totals["episodes"],
        "datasets": datasets,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Verified {len(datasets)} configurations: delete {totals['delete']:,}/"
        f"{totals['episodes']:,}; keep {totals['keep']:,}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/data/wudi/InfiData-personal/quality_runs/all_rlds"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/wudi/InfiData-personal/quality_runs/final_selection/all_rlds"),
    )
    parser.add_argument("--expect-datasets", type=int, default=44)
    parser.add_argument("--expect-episodes", type=int, default=EXPECTED_EPISODES)
    parser.add_argument("--expect-delete", type=int, default=EXPECTED_DELETE)
    parser.add_argument(
        "--robomind-decisions",
        type=Path,
        default=Path(
            "/data/wudi/InfiData-personal/quality_runs/final_selection/"
            "RoboMIND_full/episode_decisions.jsonl"
        ),
        help="Previously validated P3-P5 decisions.",
    )
    return parser


if __name__ == "__main__":
    build(build_parser().parse_args())
