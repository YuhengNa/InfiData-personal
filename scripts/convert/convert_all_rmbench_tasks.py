import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


DEFAULT_ALL_EPISODES_LIMIT = 1_000_000_000


def discover_tasks(rmbench_root: Path, task_config: str) -> list[str]:
    tasks = []
    for task_dir in sorted(p for p in rmbench_root.iterdir() if p.is_dir()):
        task_root = task_dir / task_config
        if (task_root / "data").is_dir() and (task_root / "language_annotation.json").is_file():
            tasks.append(task_dir.name)
    return tasks


def parse_task_list(values: list[str] | None) -> set[str] | None:
    if not values:
        return None

    tasks = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                tasks.add(item)
    return tasks


def build_command(
    converter_path: Path,
    rmbench_root: Path,
    out_root: Path,
    task_name: str,
    task_config: str,
    num_episodes: int,
    fps: int,
    strict: bool,
    no_export_videos: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(converter_path),
        "--rmbench_root",
        str(rmbench_root),
        "--task_name",
        task_name,
        "--task_config",
        task_config,
        "--out_root",
        str(out_root / task_name),
        "--num_episodes",
        str(num_episodes),
        "--fps",
        str(fps),
    ]
    if strict:
        cmd.append("--strict")
    if no_export_videos:
        cmd.append("--no_export_videos")
    return cmd


def write_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert all RMBench tasks under data/*/demo_clean into InfiData format."
    )
    parser.add_argument("--rmbench_root", type=str, required=True, help="RMBench data root, e.g. /path/to/RMBench/data")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for all converted InfiData tasks")
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Max episodes per task. Omit this to convert all available episodes.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task names or comma-separated task names")
    parser.add_argument("--exclude_tasks", nargs="*", default=None, help="Optional task names to skip")
    parser.add_argument("--strict", action="store_true", help="Fail inside a task on the first invalid episode")
    parser.add_argument(
        "--keep_going",
        action="store_true",
        help="Continue with remaining tasks if one task conversion fails",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip tasks whose output meta/episodes.jsonl already exists",
    )
    parser.add_argument("--no_export_videos", action="store_true")
    args = parser.parse_args()

    rmbench_root = Path(args.rmbench_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not rmbench_root.is_dir():
        raise FileNotFoundError(f"RMBench root not found: {rmbench_root}")

    include_tasks = parse_task_list(args.tasks)
    exclude_tasks = parse_task_list(args.exclude_tasks) or set()
    all_tasks = discover_tasks(rmbench_root, args.task_config)
    if include_tasks is not None:
        missing = sorted(include_tasks - set(all_tasks))
        if missing:
            raise ValueError(f"Requested tasks not found under {rmbench_root}: {missing}")
        tasks = sorted(include_tasks)
    else:
        tasks = all_tasks
    tasks = [task for task in tasks if task not in exclude_tasks]
    if not tasks:
        raise RuntimeError(f"No RMBench tasks found under {rmbench_root} with config {args.task_config!r}")

    num_episodes = args.num_episodes if args.num_episodes is not None else DEFAULT_ALL_EPISODES_LIMIT
    if num_episodes <= 0:
        raise ValueError("--num_episodes must be positive when provided")

    converter_path = Path(__file__).with_name("convert_rmbench_mini.py")
    summary = {
        "rmbench_root": str(rmbench_root),
        "out_root": str(out_root),
        "task_config": args.task_config,
        "num_episodes_per_task": None if args.num_episodes is None else args.num_episodes,
        "fps": args.fps,
        "strict": args.strict,
        "no_export_videos": args.no_export_videos,
        "total_tasks": len(tasks),
        "converted": [],
        "skipped_existing": [],
        "failed": [],
    }

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "batch_convert_summary.json"

    print(f"[INFO] RMBench root: {rmbench_root}")
    print(f"[INFO] Output root: {out_root}")
    print(f"[INFO] Found {len(tasks)} tasks")
    print(f"[INFO] Episodes per task: {'all' if args.num_episodes is None else args.num_episodes}")

    for task_name in tqdm(tasks, desc="Converting RMBench tasks"):
        task_out = out_root / task_name
        if args.skip_existing and (task_out / "meta" / "episodes.jsonl").exists():
            summary["skipped_existing"].append(task_name)
            write_summary(summary_path, summary)
            continue

        cmd = build_command(
            converter_path=converter_path,
            rmbench_root=rmbench_root,
            out_root=out_root,
            task_name=task_name,
            task_config=args.task_config,
            num_episodes=num_episodes,
            fps=args.fps,
            strict=args.strict,
            no_export_videos=args.no_export_videos,
        )

        print("\n[RUN] " + " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            summary["converted"].append(task_name)
        else:
            failure = {"task": task_name, "returncode": result.returncode}
            summary["failed"].append(failure)
            write_summary(summary_path, summary)
            if not args.keep_going:
                raise RuntimeError(f"Failed to convert task {task_name}. See command output above.")

        write_summary(summary_path, summary)

    print("\n[DONE] Batch RMBench conversion finished.")
    print(f"Converted tasks: {len(summary['converted'])}")
    print(f"Skipped existing tasks: {len(summary['skipped_existing'])}")
    print(f"Failed tasks: {len(summary['failed'])}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
