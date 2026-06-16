import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_task_roots(root: Path, recursive: bool) -> list[Path]:
    if (root / "meta" / "episodes.jsonl").exists():
        return [root]

    pattern = "**/meta/episodes.jsonl" if recursive else "*/meta/episodes.jsonl"
    return sorted({path.parent.parent for path in root.glob(pattern)})


def compact_ranges(indices: list[int]) -> list[tuple[int, int]]:
    ranges = []
    start = None
    previous = None
    for index in indices:
        if start is None:
            start = previous = index
        elif index == previous + 1:
            previous = index
        else:
            ranges.append((start, previous))
            start = previous = index
    if start is not None:
        ranges.append((start, previous))
    return ranges


def format_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(f"{start}-{end}" if start != end else str(start) for start, end in ranges)


def format_segments(segments: list[tuple[int, int]]) -> str:
    return ",".join(f"{start}-{end}" for start, end in segments)


def scan_task(task_root: Path) -> tuple[dict, list[dict]]:
    episodes_path = task_root / "meta" / "episodes.jsonl"
    memory_path = task_root / "meta" / "memory_segments.jsonl"
    task_name = task_root.name

    episodes = {}
    for item in read_jsonl(episodes_path):
        episode_index = int(item["episode_index"])
        if "num_frames" not in item:
            raise KeyError(f"Missing num_frames in {episodes_path}, episode {episode_index}")
        episodes[episode_index] = int(item["num_frames"])

    memory_segments = defaultdict(list)
    if memory_path.exists():
        for item in read_jsonl(memory_path):
            episode_index = int(item["episode_index"])
            start_frame = int(item["start_frame"])
            end_frame = int(item["end_frame"])
            memory_segments[episode_index].append((start_frame, end_frame))

    rows = []
    full_coverage = 0
    total_missing_frames = 0
    episodes_without_memory_file = 0

    for episode_index, num_frames in sorted(episodes.items()):
        covered = [False] * num_frames
        segments = sorted(memory_segments.get(episode_index, []))

        for start_frame, end_frame in segments:
            for frame_index in range(max(0, start_frame), min(num_frames - 1, end_frame) + 1):
                covered[frame_index] = True

        missing = [frame_index for frame_index, is_covered in enumerate(covered) if not is_covered]
        missing_ranges = compact_ranges(missing)
        missing_count = len(missing)
        total_missing_frames += missing_count
        if not segments:
            episodes_without_memory_file += 1
        if missing_count == 0:
            full_coverage += 1

        rows.append(
            {
                "task": task_name,
                "task_root": str(task_root),
                "episode_index": episode_index,
                "num_frames": num_frames,
                "memory_segments": format_segments(segments),
                "missing_frames": missing_count,
                "missing_ranges": format_ranges(missing_ranges),
                "is_full_coverage": missing_count == 0,
            }
        )

    summary = {
        "task": task_name,
        "task_root": str(task_root),
        "episodes": len(episodes),
        "full_coverage_episodes": full_coverage,
        "missing_coverage_episodes": len(episodes) - full_coverage,
        "total_missing_frames": total_missing_frames,
        "episodes_without_memory_segments": episodes_without_memory_file,
        "has_memory_segments_file": memory_path.exists(),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, summaries: list[dict], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summaries": summaries, "episodes": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(summaries: list[dict], rows: list[dict], only_missing: bool):
    total_tasks = len(summaries)
    total_episodes = sum(item["episodes"] for item in summaries)
    full = sum(item["full_coverage_episodes"] for item in summaries)
    missing = sum(item["missing_coverage_episodes"] for item in summaries)
    missing_frames = sum(item["total_missing_frames"] for item in summaries)

    print("=== Overall ===")
    print(f"Tasks: {total_tasks}")
    print(f"Episodes: {total_episodes}")
    print(f"Full memory coverage episodes: {full}")
    print(f"Missing memory coverage episodes: {missing}")
    print(f"Total missing frames: {missing_frames}")
    print()

    print("=== Per Task ===")
    print("task\tepisodes\tfull\tmissing_eps\tmissing_frames\tmemory_file")
    for item in summaries:
        print(
            f"{item['task']}\t{item['episodes']}\t{item['full_coverage_episodes']}\t"
            f"{item['missing_coverage_episodes']}\t{item['total_missing_frames']}\t"
            f"{'yes' if item['has_memory_segments_file'] else 'no'}"
        )

    detail_rows = [row for row in rows if (not only_missing or row["missing_frames"] > 0)]
    if not detail_rows:
        return

    print()
    print("=== Episode Detail ===")
    print("task\tepisode\tnum_frames\tmemory_segments\tmissing_frames\tmissing_ranges")
    for row in detail_rows:
        print(
            f"{row['task']}\t{row['episode_index']}\t{row['num_frames']}\t"
            f"{row['memory_segments'] or '-'}\t{row['missing_frames']}\t{row['missing_ranges'] or '-'}"
        )


def main():
    parser = argparse.ArgumentParser(description="Scan InfiData RMBench memory segment frame coverage.")
    parser.add_argument("--root", required=True, help="InfiData task root or a root containing multiple task directories")
    parser.add_argument("--recursive", action="store_true", help="recursively find task roots under --root")
    parser.add_argument("--show_all", action="store_true", help="print episodes with full coverage too")
    parser.add_argument("--output_csv", default=None, help="optional CSV path for per-episode details")
    parser.add_argument("--output_json", default=None, help="optional JSON path for summaries and details")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    task_roots = find_task_roots(root, args.recursive)
    if not task_roots:
        raise FileNotFoundError(f"No task roots with meta/episodes.jsonl found under {root}")

    summaries = []
    rows = []
    for task_root in task_roots:
        summary, task_rows = scan_task(task_root)
        summaries.append(summary)
        rows.extend(task_rows)

    print_summary(summaries, rows, only_missing=not args.show_all)

    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
        print(f"\nWrote CSV: {Path(args.output_csv).resolve()}")
    if args.output_json:
        write_json(Path(args.output_json), summaries, rows)
        print(f"Wrote JSON: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
