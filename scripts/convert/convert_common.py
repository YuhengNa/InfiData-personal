import json
import os
import re
import shutil
from pathlib import Path

import numpy as np


def speed_bin_from_T(T: int, bin_size: int = 500) -> int:
    return int(round(T / bin_size) * bin_size)


def episode_chunk(episode_index: int, chunk_size: int = 1000) -> int:
    return episode_index // chunk_size


def safe_symlink_or_copy(
    src: Path,
    dst: Path,
    copy_video: bool = False,
    overwrite: bool = False,
) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if copy_video:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def vector_stats(state: np.ndarray, action: np.ndarray) -> dict:
    return {
        "num_frames": int(len(state)),
        "state_min": state.min(axis=0).tolist(),
        "state_max": state.max(axis=0).tolist(),
        "state_mean": state.mean(axis=0).tolist(),
        "state_std": state.std(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": action.std(axis=0).tolist(),
    }


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "unknown"
