from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


PALETTE = [
    (66, 135, 245),
    (80, 180, 120),
    (245, 166, 66),
    (180, 100, 220),
    (230, 90, 90),
    (80, 190, 210),
    (180, 180, 80),
]


def _shorten(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _active_record(records: List[Dict[str, Any]], frame_idx: int) -> Dict[str, Any]:
    for rec in records:
        if int(rec.get("start_frame", 0)) <= frame_idx <= int(rec.get("end_frame", -1)):
            return rec
    return {}


def _draw_timeline(
    img,
    records: List[Dict[str, Any]],
    y: int,
    h: int,
    x0: int,
    x1: int,
    nframes: int,
    text_key: str,
):
    width = max(1, x1 - x0)
    cv2.rectangle(img, (x0, y), (x1, y + h), (45, 45, 45), -1)

    for idx, rec in enumerate(records):
        s = int(rec.get("start_frame", 0))
        e = int(rec.get("end_frame", s))
        sx = x0 + int(width * max(0, min(s, nframes - 1)) / max(1, nframes - 1))
        ex = x0 + int(width * max(0, min(e, nframes - 1)) / max(1, nframes - 1))
        ex = max(sx + 2, ex)
        color = PALETTE[idx % len(PALETTE)]
        cv2.rectangle(img, (sx, y), (ex, y + h), color, -1)
        label = _shorten(str(rec.get(text_key, "")), max(8, (ex - sx) // 7))
        if label:
            cv2.putText(img, label, (sx + 4, y + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def render_annotation_video(
    video_path: str,
    output_path: str,
    subtask_records: List[Dict[str, Any]],
    memory_records: List[Dict[str, Any]],
    panel_height: int = 180,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for visualization: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height + panel_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create visualization video: {output_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Work in RGB for visualization colors, then convert back for OpenCV's
        # BGR VideoWriter. This keeps the exported mp4 from showing swapped
        # red/blue channels in common video players.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas = np.zeros((height + panel_height, width, 3), dtype=np.uint8)
        canvas[:height, :, :] = frame_rgb
        panel = canvas[height:, :, :]
        panel[:, :, :] = (18, 18, 18)

        bar_x0 = 110
        bar_x1 = width - 12
        sub_y = height + 22
        mem_y = height + 72

        cv2.putText(canvas, "Subtask", (12, sub_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Memory", (12, mem_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        _draw_timeline(canvas, subtask_records, sub_y, 30, bar_x0, bar_x1, nframes, "subtask")
        _draw_timeline(canvas, memory_records, mem_y, 34, bar_x0, bar_x1, nframes, "summary")

        cursor_x = bar_x0 + int((bar_x1 - bar_x0) * frame_idx / max(1, nframes - 1))
        cv2.line(canvas, (cursor_x, height + 12), (cursor_x, height + panel_height - 12), (255, 255, 255), 2)

        active_subtask = _active_record(subtask_records, frame_idx).get("subtask", "")
        active_memory = _active_record(memory_records, frame_idx).get("summary", "")
        cv2.putText(canvas, f"Current subtask: {_shorten(active_subtask, 120)}", (12, height + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Current memory: {_shorten(active_memory, 120)}", (12, height + 154), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    cap.release()
    writer.release()
