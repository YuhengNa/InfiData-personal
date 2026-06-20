from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def append_skip(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def ffmpeg_read_frames(path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    if not frame_indices:
        return []
    ordered = [int(index) for index in frame_indices]
    if min(ordered) < 0:
        raise ValueError(f"Negative frame index for {path}: {min(ordered)}")

    width, height = _ffprobe_video_shape(path)
    unique = sorted(set(ordered))
    if unique == list(range(unique[0], unique[-1] + 1)):
        select_expr = f"between(n\\,{unique[0]}\\,{unique[-1]})"
    else:
        select_expr = "+".join(f"eq(n\\,{index})" for index in unique)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "0",
        "-frames:v",
        str(len(unique)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr}")

    frame_size = width * height * 3
    expected = frame_size * len(unique)
    if len(proc.stdout) != expected:
        raise RuntimeError(
            f"ffmpeg returned {len(proc.stdout)} bytes for {path}; expected {expected} "
            f"for {len(unique)} frame(s) of shape ({height}, {width}, 3)"
        )

    raw = np.frombuffer(proc.stdout, dtype=np.uint8).reshape((len(unique), height, width, 3)).copy()
    by_index = {index: raw[i] for i, index in enumerate(unique)}
    return [by_index[index] for index in ordered]


def ffmpeg_read_clip(path: Path, start_frame: int, length: int) -> list[np.ndarray]:
    return ffmpeg_read_frames(path, list(range(int(start_frame), int(start_frame) + int(length))))


def _ffprobe_video_shape(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe failed for {path}: {stderr}")
    info = json.loads(proc.stdout.decode("utf-8"))
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"])
