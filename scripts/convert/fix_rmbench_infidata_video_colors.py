import argparse
import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


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


def find_video_paths(infidata_root: Path) -> list[Path]:
    rel_paths = set()
    for item in read_jsonl(infidata_root / "meta" / "episodes.jsonl"):
        video_paths = item.get("video_paths")
        if isinstance(video_paths, dict):
            for value in video_paths.values():
                if isinstance(value, str) and value.strip():
                    rel_paths.add(Path(value))

    if rel_paths:
        return sorted(infidata_root / path for path in rel_paths)

    return sorted((infidata_root / "videos").glob("**/*.mp4"))


def copy_non_video_tree(src_root: Path, dst_root: Path):
    for src_path in src_root.rglob("*"):
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        if rel_path.parts and rel_path.parts[0] == "videos":
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst_path.hardlink_to(src_path)
        except OSError:
            shutil.copy2(src_path, dst_path)


def swap_red_blue_video(src_path: Path, dst_path: Path, overwrite: bool):
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Output video already exists: {dst_path}")

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size for {src_path}: {width}x{height}")

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp.mp4")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer: {tmp_path}")

    frames = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            fixed_bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            writer.write(fixed_bgr)
            frames += 1
    finally:
        cap.release()
        writer.release()

    if frames == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"No frames decoded from {src_path}")

    if dst_path.exists():
        dst_path.unlink()
    tmp_path.replace(dst_path)


def repair_to_output_root(infidata_root: Path, output_root: Path, overwrite: bool):
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    copy_non_video_tree(infidata_root, output_root)
    video_paths = find_video_paths(infidata_root)
    if not video_paths:
        raise FileNotFoundError(f"No mp4 videos found under {infidata_root}")

    for src_path in tqdm(video_paths, desc="Repairing videos"):
        if not src_path.exists():
            raise FileNotFoundError(f"Video listed in metadata does not exist: {src_path}")
        rel_path = src_path.relative_to(infidata_root)
        swap_red_blue_video(src_path, output_root / rel_path, overwrite=True)

    return len(video_paths)


def repair_in_place(infidata_root: Path, backup_suffix: str):
    video_paths = find_video_paths(infidata_root)
    if not video_paths:
        raise FileNotFoundError(f"No mp4 videos found under {infidata_root}")

    for path in tqdm(video_paths, desc="Repairing videos in place"):
        if not path.exists():
            raise FileNotFoundError(f"Video listed in metadata does not exist: {path}")
        backup_path = path.with_name(path.name + backup_suffix)
        if backup_path.exists():
            raise FileExistsError(f"Backup already exists, refusing to overwrite: {backup_path}")
        path.rename(backup_path)
        try:
            swap_red_blue_video(backup_path, path, overwrite=True)
        except Exception:
            if path.exists():
                path.unlink()
            backup_path.rename(path)
            raise

    return len(video_paths)


def main():
    parser = argparse.ArgumentParser(description="Fix red/blue channel swapped RMBench InfiData videos.")
    parser.add_argument("--infidata_root", required=True, help="existing InfiData task root")
    parser.add_argument("--output_root", default=None, help="write a repaired copy to this root")
    parser.add_argument("--in_place", action="store_true", help="repair videos in place and keep backups")
    parser.add_argument("--backup_suffix", default=".color_bug_backup", help="backup suffix used with --in_place")
    parser.add_argument("--overwrite", action="store_true", help="replace --output_root if it already exists")
    args = parser.parse_args()

    infidata_root = Path(args.infidata_root).resolve()
    if not infidata_root.exists():
        raise FileNotFoundError(f"InfiData root not found: {infidata_root}")
    if bool(args.output_root) == bool(args.in_place):
        raise ValueError("Choose exactly one mode: --output_root or --in_place")

    if args.in_place:
        count = repair_in_place(infidata_root, args.backup_suffix)
        print(f"\n[DONE] Repaired {count} videos in place.")
        print(f"Backups use suffix: {args.backup_suffix}")
    else:
        output_root = Path(args.output_root).resolve()
        count = repair_to_output_root(infidata_root, output_root, overwrite=args.overwrite)
        print(f"\n[DONE] Repaired {count} videos.")
        print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
