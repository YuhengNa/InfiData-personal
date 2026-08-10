#!/usr/bin/env python3
"""Apply the confirmed RoboMIND quality policy to completed S1/S2/S3 runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


MASTER_ACTION = "absolute_master_joint_position"
TIANKUNG_ARM_DIMS = set(range(7)) | set(range(19, 26))


def _flagged_s3(episode: dict[str, Any]) -> bool:
    return bool(episode["s3_state"]["flagged"] or episode["s3_action"]["flagged"])


def _hit_frames(episode: dict[str, Any], dimensions: set[int] | None = None) -> set[int]:
    frames = set()
    for signal in ("s1_state", "s1_action"):
        for hit in episode[signal]["hits"]:
            if dimensions is None or hit["dim"] in dimensions:
                frames.add(hit["frame"])
    return frames


def apply_policy(episode: dict[str, Any]) -> dict[str, Any]:
    metadata = episode["metadata"]
    action_representation = metadata.get("action_representation", "")
    state_dim = len(episode["s1_state"].get("scales", {}).get("residual", []))
    action_dim = len(episode["s1_action"].get("scales", {}).get("residual", []))

    if action_representation == MASTER_ACTION:
        group = "master_puppet"
        s1 = bool(episode["s1_state"]["flagged"])
        s2 = False
        detail = {"s1_source": "state_only", "s2_source": "disabled"}
    elif state_dim == action_dim == 38 and metadata.get("robot_type") == "tienkung_humanoid":
        group = "tienkung_s38"
        s1 = bool(_hit_frames(episode, TIANKUNG_ARM_DIMS))
        s2 = any(
            item.get("flagged", False)
            and item["dim"] in TIANKUNG_ARM_DIMS
            and item.get("active_samples", 0) >= 20
            for item in episode["s2"]["dimensions"]
        )
        detail = {
            "s1_s2_dimensions": "arms_only:0-6,19-25",
            "s2_min_active_samples": 20,
        }
    else:
        group = "franka_sim"
        s1_frames = _hit_frames(episode)
        s1_rate = len(s1_frames) / episode["num_frames"] if episode["num_frames"] else 0.0
        s1 = len(s1_frames) >= 3 and s1_rate >= 0.01
        s2 = bool(episode["s2"]["flagged"])
        detail = {
            "s1_hit_frames": len(s1_frames),
            "s1_hit_rate": s1_rate,
            "s1_min_frames": 3,
            "s1_min_rate": 0.01,
        }

    s3 = _flagged_s3(episode)
    rules = [rule for rule, flagged in (("S1", s1), ("S2", s2), ("S3", s3)) if flagged]
    return {"group": group, "delete": bool(rules), "rules": rules, **detail}


def run(args: argparse.Namespace) -> None:
    result_files = sorted(args.runs_root.glob("*/**/episodes.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No episodes.jsonl found under {args.runs_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_tmp = args.output_dir / "episode_decisions.jsonl.tmp"
    candidates_tmp = args.output_dir / "delete_candidates.jsonl.tmp"
    datasets = []
    total = Counter()

    with decisions_tmp.open("w") as decisions, candidates_tmp.open("w") as candidates:
        for path in result_files:
            dataset = path.relative_to(args.runs_root).parts[0]
            counts = Counter()
            with path.open() as source:
                for line in source:
                    episode = json.loads(line)
                    result = apply_policy(episode)
                    record = {
                        "dataset": dataset,
                        "episode": episode["episode"],
                        "num_frames": episode["num_frames"],
                        "delete": result["delete"],
                        "rules": result["rules"],
                        "group": result["group"],
                        "parquet_path": episode["metadata"].get("parquet_path"),
                        "details": {k: v for k, v in result.items()
                                    if k not in {"delete", "rules", "group"}},
                    }
                    text = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    decisions.write(text)
                    if result["delete"]:
                        candidates.write(text)
                    counts["episodes"] += 1
                    counts["delete"] += result["delete"]
                    for rule in result["rules"]:
                        counts[rule] += 1

            datasets.append({
                "dataset": dataset,
                **counts,
                "delete_rate": counts["delete"] / counts["episodes"],
            })
            total.update(counts)
            print(f"{dataset}: {counts['delete']:,}/{counts['episodes']:,} "
                  f"({100 * counts['delete'] / counts['episodes']:.2f}%)")

    decisions_tmp.replace(args.output_dir / "episode_decisions.jsonl")
    candidates_tmp.replace(args.output_dir / "delete_candidates.jsonl")
    summary = {
        "policy": {
            "master_puppet": "S1 state only; S2 disabled; S3 retained",
            "franka_sim": "S1 >= 3 unique frames and >= 1% frames; S2/S3 retained",
            "tienkung_s38": "S1/S2 arms 0-6,19-25 only; S2 active_samples >= 20; S3 retained",
            "source_data_modified": False,
        },
        "total": {
            **total,
            "delete_rate": total["delete"] / total["episodes"],
        },
        "datasets": datasets,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"TOTAL: {total['delete']:,}/{total['episodes']:,} "
          f"({100 * total['delete'] / total['episodes']:.2f}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/data/wudi/InfiData-personal/quality_runs/all_rlds/RoboMIND_full"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/wudi/InfiData-personal/quality_runs/final_selection/RoboMIND_full"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
