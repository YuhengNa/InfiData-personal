#!/usr/bin/env python3
"""Screen RLDS episodes with the Qwen S1/S2/S3 state-action checks.

The tool is deliberately non-destructive: it writes a review manifest and
plots, but never modifies the source RLDS/TFDS dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from tqdm import tqdm


@dataclass
class Episode:
    key: str
    state: np.ndarray
    action: np.ndarray
    metadata: dict[str, Any]


@dataclass
class S1Config:
    median_kernel: int = 5
    savgol_window: int = 9
    savgol_polyorder: int = 2
    residual_z: float = 8.0
    acceleration_z: float = 8.0
    jerk_z: float = 8.0
    scale_floor: float = 1e-6
    relative_scale_floor: float = 0.002
    max_samples: int = 1_000_000


@dataclass
class S2Config:
    da_threshold: float = 0.6
    max_lag: int = 10
    min_active: int = 10
    motion_epsilon: float = 1e-5
    flag_negative_lag: bool = True


@dataclass
class S3Config:
    alpha: float = 1.5
    q_low: float = 0.01
    q_high: float = 0.99
    max_samples: int = 1_000_000


def _odd_at_most(value: int, length: int, minimum: int = 3) -> int:
    value = min(value, length if length % 2 else length - 1)
    if value % 2 == 0:
        value -= 1
    return value if value >= minimum else 0


def _fill_nonfinite(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    x = np.arange(len(out))
    for dim in range(out.shape[1]):
        good = np.isfinite(out[:, dim])
        if good.all():
            continue
        if not good.any():
            out[:, dim] = 0.0
        elif good.sum() == 1:
            out[:, dim] = out[good, dim][0]
        else:
            out[~good, dim] = np.interp(x[~good], x[good], out[good, dim])
    return out


def smooth(values: np.ndarray, median_kernel: int = 5, savgol_window: int = 9,
           polyorder: int = 2) -> np.ndarray:
    values = _fill_nonfinite(np.atleast_2d(values))
    if values.shape[0] < 3:
        return values.copy()
    kernel = _odd_at_most(median_kernel, values.shape[0])
    filtered = median_filter(values, size=(kernel, 1), mode="nearest") if kernel else values
    window = _odd_at_most(savgol_window, values.shape[0])
    if not window:
        return filtered
    order = min(polyorder, window - 1)
    return savgol_filter(filtered, window_length=window, polyorder=order, axis=0, mode="interp")


def _robust_z(metric: np.ndarray, floor: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric = np.asarray(metric, dtype=np.float64)
    center = np.nanmedian(metric, axis=0)
    mad = np.nanmedian(np.abs(metric - center), axis=0)
    scale = np.maximum(1.4826 * mad, floor)
    return (metric - center) / scale, center, scale


def _robust_center_scale(metric: np.ndarray,
                         floor: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit robust upper-tail parameters without allocating a full z-score array."""
    metric = np.asarray(metric)
    center = np.nanmedian(metric, axis=0)
    mad = np.nanmedian(np.abs(metric - center), axis=0)
    return center, np.maximum(1.4826 * mad, floor)


def _s1_metrics(values: np.ndarray, cfg: S1Config) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.asarray(values, dtype=np.float64)
    clean = _fill_nonfinite(raw)
    baseline = smooth(clean, cfg.median_kernel, cfg.savgol_window, cfg.savgol_polyorder)
    residual = np.abs(clean - baseline)

    acceleration = np.full_like(clean, np.nan)
    if len(clean) >= 3:
        acceleration[1:-1] = np.abs(clean[2:] - 2.0 * clean[1:-1] + clean[:-2])

    jerk = np.full_like(clean, np.nan)
    if len(clean) >= 5:
        jerk[2:-2] = np.abs(
            clean[4:] - 2.0 * clean[3:-1] + 2.0 * clean[1:-3] - clean[:-4]
        ) / 2.0
    return baseline, {"residual": residual, "acceleration": acceleration, "jerk": jerk}


def fit_s1(episodes: Iterable[Episode], cfg: S1Config, signal: str) -> dict[str, Any]:
    values_chunks, metric_chunks = [], {"residual": [], "acceleration": [], "jerk": []}
    for episode in tqdm(episodes, desc=f"Fitting S1 {signal}", unit="episode"):
        values = episode.state if signal == "state" else episode.action
        _, metrics = _s1_metrics(values, cfg)
        values_chunks.append(_fill_nonfinite(values))
        for name, metric in metrics.items():
            metric_chunks[name].append(metric)
    values = np.concatenate(values_chunks, axis=0)
    metrics = {name: np.concatenate(chunks, axis=0) for name, chunks in metric_chunks.items()}
    if len(values) > cfg.max_samples:
        keep = np.random.default_rng(0).choice(len(values), cfg.max_samples, replace=False)
        values = values[keep]
        metrics = {name: metric[keep] for name, metric in metrics.items()}
    return _fit_s1_sample(values, metrics, cfg, signal, total_rows=sum(
        len(chunk) for chunk in values_chunks
    ))


def _fit_s1_sample(values: np.ndarray, metrics: dict[str, np.ndarray], cfg: S1Config,
                   signal: str, total_rows: int) -> dict[str, Any]:
    signal_span = np.nanquantile(values, 0.99, axis=0) - np.nanquantile(values, 0.01, axis=0)
    floor = np.maximum(cfg.scale_floor, cfg.relative_scale_floor * signal_span)
    result: dict[str, Any] = {
        "signal": signal, "total_rows": int(total_rows), "sample_rows": len(values),
    }
    for name, metric in metrics.items():
        center, scale = _robust_center_scale(metric, floor)
        z_threshold = getattr(cfg, f"{name}_z")
        result[name] = {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "threshold": (center + z_threshold * scale).tolist(),
        }
    return result


def detect_s1(values: np.ndarray, cfg: S1Config,
              thresholds: dict[str, Any] | None = None,
              ignored_dimensions: set[int] | None = None) -> dict[str, Any]:
    """Return sudden-change frames for one [time, dimension] signal."""
    raw = np.asarray(values, dtype=np.float64)
    baseline, metrics = _s1_metrics(raw, cfg)
    if thresholds is None:
        clean = _fill_nonfinite(raw)
        span = np.nanquantile(clean, 0.99, axis=0) - np.nanquantile(clean, 0.01, axis=0)
        floor = np.maximum(cfg.scale_floor, cfg.relative_scale_floor * span)
        computed: dict[str, Any] = {}
        for name, metric in metrics.items():
            z, center, scale = _robust_z(metric, floor)
            z_threshold = getattr(cfg, f"{name}_z")
            computed[name] = {
                "z": z,
                "center": center,
                "scale": scale,
                "threshold": center + z_threshold * scale,
            }
    else:
        computed = {}
        for name, metric in metrics.items():
            center = np.asarray(thresholds[name]["center"])
            scale = np.asarray(thresholds[name]["scale"])
            computed[name] = {
                "z": (metric - center) / scale,
                "center": center,
                "scale": scale,
                "threshold": np.asarray(thresholds[name]["threshold"]),
            }
    residual, acceleration, jerk = (metrics[name] for name in ("residual", "acceleration", "jerk"))
    rz, az, jz = (computed[name]["z"] for name in ("residual", "acceleration", "jerk"))
    rscale, ascale, jscale = (computed[name]["scale"] for name in ("residual", "acceleration", "jerk"))
    rlimit, alimit, jlimit = (
        computed[name]["threshold"] for name in ("residual", "acceleration", "jerk")
    )
    sudden_change_mask = (residual >= rlimit) & (
        (acceleration >= alimit) | (jerk >= jlimit)
    )
    ignored_dimensions = ignored_dimensions or set()
    for dim in ignored_dimensions:
        if 0 <= dim < sudden_change_mask.shape[1]:
            sudden_change_mask[:, dim] = False
    mask = sudden_change_mask | ~np.isfinite(raw)
    frames, dims = np.nonzero(mask)
    hits = []
    for frame, dim in zip(frames.tolist(), dims.tolist()):
        hits.append({
            "frame": frame,
            "dim": dim,
            "value": _json_number(raw[frame, dim]),
            "context": [_json_number(value) for value in
                        raw[max(0, frame - 2):min(len(raw), frame + 3), dim]],
            "baseline": float(baseline[frame, dim]),
            "residual": float(residual[frame, dim]),
            "residual_z": float(rz[frame, dim]),
            "acceleration_z": float(az[frame, dim]),
            "jerk_z": float(jz[frame, dim]),
        })
    return {
        "flagged": bool(hits),
        "frames": sorted(set(frames.tolist())),
        "hits": hits,
        "ignored_dimensions": sorted(ignored_dimensions),
        "scales": {
            "residual": rscale.tolist(),
            "acceleration": ascale.tolist(),
            "jerk": jscale.tolist(),
        },
    }


def _aligned(a: np.ndarray, s: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive lag means action leads state by ``lag`` frames."""
    if lag > 0:
        return a[:-lag], s[lag:]
    if lag < 0:
        return a[-lag:], s[:lag]
    return a, s


def _epsilon_sign(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Map values to {-1, 0, 1}, treating epsilon-scale motion as stationary."""
    return np.where(values > epsilon, 1, np.where(values < -epsilon, -1, 0))


def detect_s2(state: np.ndarray, action: np.ndarray, cfg: S2Config,
              action_is_delta: bool = False,
              ignored_dimensions: set[int] | None = None) -> dict[str, Any]:
    """Check trend alignment on dimensions with matching physical semantics."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    ignored_dimensions = ignored_dimensions or set()
    dims = min(state.shape[1], action.shape[1])
    if dims == 0 or len(state) < 4:
        return {
            "flagged": False,
            "reason": "insufficient_data",
            "ignored_dimensions": sorted(ignored_dimensions),
            "dimensions": [],
        }
    state = smooth(state[:, :dims])
    action = _fill_nonfinite(action[:, :dims])
    if action_is_delta:
        action = np.cumsum(action, axis=0)
    action = smooth(action)
    sd = np.diff(state, axis=0)
    ad = np.diff(action, axis=0)
    results = []
    for dim in range(dims):
        if dim in ignored_dimensions:
            results.append({"dim": dim, "evaluated": False, "reason": "ignored_dimension"})
            continue
        best: dict[str, Any] | None = None
        for lag in range(-cfg.max_lag, cfg.max_lag + 1):
            aa, ss = _aligned(ad[:, dim], sd[:, dim], lag)
            finite = np.isfinite(aa) & np.isfinite(ss)
            active = finite & ((np.abs(aa) > cfg.motion_epsilon) | (np.abs(ss) > cfg.motion_epsilon))
            if active.sum() < cfg.min_active:
                continue
            av, sv = aa[active], ss[active]
            astd, sstd = float(np.std(av)), float(np.std(sv))
            corr = float(np.corrcoef(av, sv)[0, 1]) if astd > 0 and sstd > 0 else -1.0
            a_sign = _epsilon_sign(av, cfg.motion_epsilon)
            s_sign = _epsilon_sign(sv, cfg.motion_epsilon)
            agreement = float(np.mean(a_sign == s_sign))
            candidate = {"lag": lag, "correlation": corr, "directional_agreement": agreement,
                         "active_samples": int(active.sum())}
            if best is None or (corr, agreement) > (best["correlation"], best["directional_agreement"]):
                best = candidate
        if best is None:
            results.append({"dim": dim, "evaluated": False, "reason": "insufficient_motion"})
            continue
        bad_da = best["directional_agreement"] < cfg.da_threshold
        anti_causal = cfg.flag_negative_lag and best["lag"] < 0
        results.append({"dim": dim, "evaluated": True, **best, "bad_da": bad_da,
                        "anti_causal": anti_causal, "flagged": bad_da or anti_causal})
    evaluated = [item for item in results if item.get("evaluated")]
    return {
        "flagged": any(item.get("flagged", False) for item in evaluated),
        "action_is_delta": action_is_delta,
        "ignored_dimensions": sorted(ignored_dimensions),
        "evaluated_dimensions": len(evaluated),
        "dimensions": results,
    }


def _state_action_schema(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("state_action_schema_json", {})
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def s2_is_compatible(metadata: dict[str, Any], state_dim: int, action_dim: int) -> tuple[bool, str]:
    if state_dim != action_dim:
        return False, "state_action_dimension_mismatch"
    schema = _state_action_schema(metadata)
    state_layout, action_layout = schema.get("state_layout"), schema.get("action_layout")
    if state_layout and action_layout:
        return (state_layout == action_layout,
                "matching_layout" if state_layout == action_layout else "state_action_layout_mismatch")
    state_rep = str(metadata.get("state_representation", "")).lower()
    action_rep = str(metadata.get("action_representation", "")).lower()
    comparable = any(token in state_rep and token in action_rep
                     for token in ("joint", "eef", "ee_pose", "cartesian"))
    return comparable, "matching_physical_representation" if comparable else "unknown_physical_semantics"


def parse_action_is_delta(value: Any) -> bool | None:
    """Parse explicit delta metadata without treating non-empty strings as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def resolve_action_is_delta(value: Any) -> tuple[bool, str]:
    """Resolve unknown delta metadata to this project's absolute-source default."""
    parsed = parse_action_is_delta(value)
    if parsed is None:
        return False, "default_absolute"
    return parsed, "metadata"


def _uniform_sample_rows(episodes: list[Episode], signal: str, max_samples: int,
                         rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Sample signal rows uniformly over the full dataset without replacement."""
    arrays = [
        np.asarray(episode.state if signal == "state" else episode.action, dtype=np.float64)
        for episode in episodes
    ]
    total_rows = sum(len(array) for array in arrays)
    if total_rows <= max_samples:
        return np.concatenate(arrays, axis=0), total_rows

    selected = np.sort(rng.choice(total_rows, max_samples, replace=False))
    chunks = []
    offset = 0
    for array in arrays:
        end = offset + len(array)
        left = np.searchsorted(selected, offset, side="left")
        right = np.searchsorted(selected, end, side="left")
        if right > left:
            chunks.append(array[selected[left:right] - offset])
        offset = end
    return np.concatenate(chunks, axis=0), total_rows


def fit_s3(episodes: Iterable[Episode], cfg: S3Config,
           state_grippers: set[int], action_grippers: set[int]) -> dict[str, Any]:
    episodes = list(episodes)
    rng = np.random.default_rng(0)
    output: dict[str, Any] = {}
    for name in ("state", "action"):
        values, total_rows = _uniform_sample_rows(episodes, name, cfg.max_samples, rng)
        grippers = state_grippers if name == "state" else action_grippers
        output[name] = _fit_s3_sample(values, total_rows, cfg, grippers)
    output["config"] = asdict(cfg)
    return output


def _fit_s3_sample(values: np.ndarray, total_rows: int, cfg: S3Config,
                   grippers: set[int]) -> dict[str, Any]:
    finite_values = np.where(np.isfinite(values), values, np.nan)
    finite_counts = np.isfinite(values).sum(axis=0)
    q01 = np.full(values.shape[1], np.nan, dtype=np.float64)
    q99 = np.full(values.shape[1], np.nan, dtype=np.float64)
    has_finite = finite_counts > 0
    if np.any(has_finite):
        q01[has_finite] = np.nanquantile(finite_values[:, has_finite], cfg.q_low, axis=0)
        q99[has_finite] = np.nanquantile(finite_values[:, has_finite], cfg.q_high, axis=0)
    span = q99 - q01
    lower, upper = q01 - cfg.alpha * span, q99 + cfg.alpha * span
    invalid_dimensions = np.flatnonzero(~np.isfinite(lower) | ~np.isfinite(upper))
    return {
        "q01": q01.tolist(), "q99": q99.tolist(),
        "lower": lower.tolist(), "upper": upper.tolist(),
        "gripper_indices": sorted(grippers),
        "finite_counts": finite_counts.tolist(),
        "invalid_threshold_dimensions": invalid_dimensions.tolist(),
        "total_rows": int(total_rows), "sample_rows": int(len(values)),
    }


def detect_s3(values: np.ndarray, thresholds: dict[str, Any], grippers: set[int]) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float64)
    lower, upper = np.asarray(thresholds["lower"]), np.asarray(thresholds["upper"])
    extreme_mask = (raw < lower) | (raw > upper)
    for dim in grippers:
        if 0 <= dim < extreme_mask.shape[1]:
            extreme_mask[:, dim] = False
    nonfinite_mask = ~np.isfinite(raw)
    mask = extreme_mask | nonfinite_mask
    frames, dims = np.nonzero(mask)
    hits = [{
        "frame": frame, "dim": dim, "value": _json_number(raw[frame, dim]),
        "lower": float(lower[dim]), "upper": float(upper[dim]),
        "reason": "nonfinite" if nonfinite_mask[frame, dim] else "extreme",
    } for frame, dim in zip(frames.tolist(), dims.tolist())]
    invalid_dimensions = np.flatnonzero(~np.isfinite(lower) | ~np.isfinite(upper))
    return {
        "flagged": bool(hits),
        "frames": sorted(set(frames.tolist())),
        "hits": hits,
        "invalid_threshold_dimensions": invalid_dimensions.tolist(),
    }


class _PriorityReservoir:
    """Exact uniform row sample with fixed retained memory.

    Each input row receives an independent random priority. Keeping the rows
    with the smallest priorities is equivalent to uniform sampling without
    replacement over the complete stream.
    """

    def __init__(self, capacity: int, columns: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("calibration sample capacity must be positive")
        self.capacity = capacity
        self.columns = columns
        self._values = np.empty((capacity, columns), dtype=np.float32)
        self._priorities = np.empty(capacity, dtype=np.float64)
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.columns:
            raise ValueError(f"expected calibration rows [N, {self.columns}], got {values.shape}")
        if not len(values):
            return
        priorities = self._rng.random(len(values))
        offset = 0
        if self._size < self.capacity:
            count = min(self.capacity - self._size, len(values))
            end = self._size + count
            self._values[self._size:end] = values[:count]
            self._priorities[self._size:end] = priorities[:count]
            self._size = end
            offset = count
        if offset == len(values):
            return

        priorities = priorities[offset:]
        values = values[offset:]
        candidate_rows = np.flatnonzero(priorities < self._priorities.max())
        if not len(candidate_rows):
            return
        candidate_priorities = priorities[candidate_rows]
        pool = np.concatenate((self._priorities, candidate_priorities))
        keep = np.argpartition(pool, self.capacity - 1)[:self.capacity]
        old_keep = keep[keep < self.capacity]
        new_keep = keep[keep >= self.capacity] - self.capacity
        retained = np.zeros(self.capacity, dtype=bool)
        retained[old_keep] = True
        replace = np.flatnonzero(~retained)
        self._values[replace] = values[candidate_rows[new_keep]]
        self._priorities[replace] = candidate_priorities[new_keep]

    @property
    def sample(self) -> np.ndarray:
        return self._values[:self._size]


def _calibration_bundle(episode: Episode, cfg: S1Config) -> np.ndarray:
    chunks = []
    for values in (episode.state, episode.action):
        raw = np.asarray(values, dtype=np.float64)
        clean = _fill_nonfinite(raw)
        _, metrics = _s1_metrics(raw, cfg)
        chunks.extend((clean, raw, metrics["residual"], metrics["acceleration"], metrics["jerk"]))
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)


def _signal_sample(sample: np.ndarray, offset: int, dimensions: int
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    blocks = [sample[:, offset + index * dimensions:offset + (index + 1) * dimensions]
              for index in range(5)]
    return blocks[0], blocks[1], {
        "residual": blocks[2], "acceleration": blocks[3], "jerk": blocks[4],
    }


def _sample_rows(values: np.ndarray, rows: int, seed: int) -> np.ndarray:
    if len(values) <= rows:
        return values
    keep = np.random.default_rng(seed).choice(len(values), rows, replace=False)
    return values[keep]


def _json_number(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else str(float(value))


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "numpy"):
        return _decode(value.numpy())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode(value.item())
        return value.tolist()
    if isinstance(value, dict):
        return {key: _decode(child) for key, child in value.items()}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _episode_key(metadata: dict[str, Any], ordinal: int) -> str:
    global_key = metadata.get("global_episode_key")
    if global_key is not None and str(global_key):
        return str(global_key)
    episode_index = metadata.get("episode_index")
    return str(ordinal if episode_index is None else episode_index)


def iter_rlds(dataset_dir: Path, split: str, max_episodes: int | None,
              desc: str = "Reading RLDS") -> Iterator[Episode]:
    """Read only state/action/metadata fields from TFRecords, never camera tensors."""
    import tensorflow as tf
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    if split not in builder.info.splits:
        raise ValueError(f"Unknown split {split!r}; available: {list(builder.info.splits)}")
    if str(builder.info.file_format).lower() not in {"fileformat.tfrecord", "tfrecord"}:
        raise ValueError(f"Only TFRecord-backed RLDS is supported, got {builder.info.file_format}")

    serialized = builder.info.features.get_serialized_info()
    selected = {
        "episode_metadata": serialized["episode_metadata"],
        "steps": {
            "observation": {"state": serialized["steps"]["observation"]["state"]},
            "action": serialized["steps"]["action"],
        },
    }
    parser = tfds.core.example_parser.ExampleParser(selected)
    split_info = builder.info.splits[split]
    files = [str(Path(str(builder.data_dir)) / name) for name in split_info.filenames]
    dataset = tf.data.TFRecordDataset(
        files, num_parallel_reads=1, buffer_size=1024 * 1024,
    ).map(parser.parse_example, num_parallel_calls=1, deterministic=True)
    total = split_info.num_examples
    if max_episodes is not None:
        dataset = dataset.take(max_episodes)
        total = min(total, max_episodes)
    progress = tqdm(dataset, total=total, desc=desc, unit="episode")
    for ordinal, raw in enumerate(progress):
        metadata = _decode(raw["episode_metadata"])
        state = raw["steps"]["observation"]["state"].numpy()
        action = raw["steps"]["action"].numpy()
        key = _episode_key(metadata, ordinal)
        yield Episode(key, state, action, metadata)


def _iter_episode_cache(path: Path, total: int) -> Iterator[Episode]:
    with path.open("rb") as cache_file:
        progress = tqdm(total=total, desc="Screening cached episodes", unit="episode")
        try:
            while True:
                try:
                    episode = pickle.load(cache_file)
                except EOFError:
                    break
                yield episode
                progress.update(1)
        finally:
            progress.close()


_GROUPED_DIMENSION = re.compile(r"^(.*?)(?:\[(\d+)\]|:(\d+))\s*$")


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(value) if value else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _json_sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        decoded = json.loads(value) if value else []
    except (TypeError, json.JSONDecodeError):
        return []
    return decoded if isinstance(decoded, list) else []


def _dimension_sources(metadata: dict[str, Any], signal: str) -> list[tuple[str, Any, bool]]:
    """Return ordered (source, entries, allow_prefix) dimension descriptions."""
    schema = _state_action_schema(metadata)
    sources: list[tuple[str, Any, bool]] = [
        (f"state_action_schema_json.{signal}_layout", schema.get(f"{signal}_layout"), False),
        (f"state_action_schema_json.{signal}_dims", schema.get(f"{signal}_dims"), False),
        (f"state_action_schema_json.{signal}_feature.names",
         _json_mapping(schema.get(f"{signal}_feature")).get("names"), True),
    ]

    top_fields = _json_sequence(metadata.get(f"{signal}_fields_json"))
    top_dims = _json_sequence(metadata.get(f"{signal}_dims_json"))
    if top_fields and len(top_fields) == len(top_dims) and all(
        isinstance(width, (int, np.integer)) for width in top_dims
    ):
        top_dims = [f"{name}:{int(width)}" for name, width in zip(top_fields, top_dims)]
    sources.append((f"{signal}_dims_json", top_dims, False))

    original = _json_mapping(metadata.get("original_feature_schema_json"))
    sources.append(
        (f"original_feature_schema_json.{signal}_feature.names",
         _json_mapping(original.get(f"{signal}_feature")).get("names"), True),
    )

    robots = _json_mapping(metadata.get("robots_json"))
    keys = [metadata.get("robot_schema_key"), schema.get("schema_key")]
    robot: dict[str, Any] = {}
    for key in keys:
        if key in robots and isinstance(robots[key], dict):
            robot = robots[key]
            break
    if not robot and len(robots) == 1:
        only = next(iter(robots.values()))
        robot = only if isinstance(only, dict) else {}
    robot_schema = _json_mapping(robot.get("state_action_schema"))
    sources.extend([
        (f"robots_json.state_action_schema.{signal}_layout",
         robot_schema.get(f"{signal}_layout"), False),
        (f"robots_json.state_action_schema.{signal}_dims",
         robot_schema.get(f"{signal}_dims"), False),
        (f"robots_json.state_action_schema.{signal}_feature.names",
         _json_mapping(robot_schema.get(f"{signal}_feature")).get("names"), True),
    ])
    return sources


def _expand_dimension_entries(entries: Any, dimension: int,
                              allow_prefix: bool) -> tuple[list[tuple[str, int, int]], str] | None:
    """Expand dimension descriptions into (name, start, width) blocks."""
    if isinstance(entries, dict):
        entries = [f"{name}:{width}" for name, width in entries.items()]
    if not isinstance(entries, (list, tuple)) or not entries:
        return None

    # A list with one name per physical dimension is already expanded. Brackets
    # in such names are indices, not block widths.
    if len(entries) == dimension and all(isinstance(item, str) for item in entries):
        return [(str(name), index, 1) for index, name in enumerate(entries)], "complete"

    if allow_prefix and len(entries) <= dimension and all(isinstance(item, str) for item in entries):
        status = "complete" if len(entries) == dimension else "partial"
        return [(str(name), index, 1) for index, name in enumerate(entries)], status

    blocks: list[tuple[str, int, int]] = []
    offset = 0
    for entry in entries:
        if not isinstance(entry, str):
            return None
        match = _GROUPED_DIMENSION.fullmatch(entry.strip())
        if match:
            name = match.group(1).strip()
            width = int(match.group(2) or match.group(3))
        else:
            name, width = entry.strip(), 1
        if not name or width < 1:
            return None
        blocks.append((name, offset, width))
        offset += width
    return (blocks, "complete") if offset == dimension else None


def _is_gripper_block(name: str, width: int) -> bool:
    normalized = name.strip().lower()
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if "gripper" in tokens:
        return True
    # AgiBot stores one scalar position per end effector under this exact field.
    # Generic EEF/hand pose fields are deliberately not classified as grippers.
    return normalized.endswith("/effector/position") and width <= 2


def resolve_gripper_indices(raw: str, metadata: dict[str, Any], signal: str,
                            dimension: int) -> tuple[set[int], dict[str, Any]]:
    """Resolve gripper dimensions and retain provenance for auditability."""
    if signal not in {"state", "action"}:
        raise ValueError(f"Unknown signal {signal!r}")
    if raw.strip().lower() != "auto":
        try:
            indices = {int(item) for item in raw.split(",") if item.strip()}
        except ValueError as exc:
            raise ValueError(f"Invalid --{signal}-gripper-indices value: {raw!r}") from exc
        invalid = sorted(index for index in indices if not 0 <= index < dimension)
        if invalid:
            raise ValueError(
                f"{signal} gripper indices {invalid} are outside dimension {dimension}"
            )
        return indices, {
            "status": "explicit", "source": "command_line", "indices": sorted(indices),
            "dimension": dimension,
        }

    valid: list[tuple[set[int], dict[str, Any]]] = []
    for source, entries, allow_prefix in _dimension_sources(metadata, signal):
        expanded = _expand_dimension_entries(entries, dimension, allow_prefix)
        if expanded is None:
            continue
        blocks, coverage = expanded
        indices = {
            index
            for name, start, width in blocks
            if _is_gripper_block(name, width)
            for index in range(start, start + width)
        }
        valid.append((indices, {
            "status": "resolved" if coverage == "complete" else "partial",
            "source": source,
            "coverage": coverage,
            "indices": sorted(indices),
            "dimension": dimension,
        }))

    # Prefer an explicit gripper-bearing schema over a higher-priority but less
    # informative layout. Otherwise the first dimensionally valid source wins.
    for indices, resolution in valid:
        if indices:
            return indices, resolution
    if valid:
        return valid[0]
    return set(), {
        "status": "unresolved", "source": None, "coverage": "none",
        "indices": [], "dimension": dimension,
    }


def _parse_indices(raw: str, metadata: dict[str, Any], layout_key: str,
                   dimension: int | None = None) -> set[int]:
    """Compatibility wrapper for callers that only need the resolved set."""
    signal = layout_key.removesuffix("_layout")
    if dimension is None:
        schema = _state_action_schema(metadata)
        dimension = int(schema.get(f"{signal}_dim", 0))
        if not dimension:
            layout = schema.get(layout_key, [])
            dimension = len(layout) if isinstance(layout, list) else 0
    return resolve_gripper_indices(raw, metadata, signal, dimension)[0]


def _plot_review(episode: Episode, report: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for dim in range(episode.state.shape[1]):
        axes[0].plot(episode.state[:, dim], lw=0.7, label=f"s{dim}")
    for dim in range(episode.action.shape[1]):
        axes[1].plot(episode.action[:, dim], lw=0.7, label=f"a{dim}")
    flagged = sorted(set(report["s1_state"]["frames"] + report["s1_action"]["frames"]
                         + report["s3_state"]["frames"] + report["s3_action"]["frames"]))
    for axis in axes:
        for frame in flagged[:200]:
            axis.axvline(frame, color="red", alpha=0.18, lw=0.7)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("state")
    axes[1].set_ylabel("action")
    axes[1].set_xlabel("frame")
    axes[0].set_title(f"episode {episode.key}; rules={','.join(report['failed_rules'])}")
    if episode.state.shape[1] <= 16:
        axes[0].legend(ncol=7, fontsize=7)
        axes[1].legend(ncol=7, fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _fit_sampled_thresholds(sample: np.ndarray, total_rows: int, state_dim: int,
                            action_dim: int, s1cfg: S1Config, s3cfg: S3Config,
                            state_grippers: set[int], action_grippers: set[int]
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
    offsets = {"state": (0, state_dim), "action": (5 * state_dim, action_dim)}
    s1_rows = _sample_rows(sample, s1cfg.max_samples, seed=1)
    s1_thresholds = {}
    for signal, (offset, dimensions) in offsets.items():
        clean, _, metrics = _signal_sample(s1_rows, offset, dimensions)
        s1_thresholds[signal] = _fit_s1_sample(
            clean, metrics, s1cfg, signal, total_rows,
        )

    s3_rows = _sample_rows(sample, s3cfg.max_samples, seed=2)
    s3: dict[str, Any] = {}
    for signal, (offset, dimensions) in offsets.items():
        _, raw, _ = _signal_sample(s3_rows, offset, dimensions)
        grippers = state_grippers if signal == "state" else action_grippers
        s3[signal] = _fit_s3_sample(raw, total_rows, s3cfg, grippers)
    s3["config"] = asdict(s3cfg)
    return s1_thresholds, s3


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    s1cfg = S1Config(residual_z=args.s1_residual_z, acceleration_z=args.s1_acceleration_z,
                     jerk_z=args.s1_jerk_z, relative_scale_floor=args.s1_relative_scale_floor,
                     max_samples=args.s1_max_samples)
    s2cfg = S2Config(da_threshold=args.s2_da_threshold, max_lag=args.s2_max_lag,
                     min_active=args.s2_min_active, flag_negative_lag=not args.s2_allow_negative_lag)
    s3cfg = S3Config(alpha=args.s3_alpha, max_samples=args.s3_max_samples)
    if args.calibration_memory_mb < 1:
        raise ValueError("--calibration-memory-mb must be positive")
    if s1cfg.max_samples < 1 or s3cfg.max_samples < 1:
        raise ValueError("--s1-max-samples and --s3-max-samples must be positive")

    cache_path = output / ".state_action_episodes.pkl"
    partial_cache = output / ".state_action_episodes.pkl.partial"
    cache_path.unlink(missing_ok=True)
    partial_cache.unlink(missing_ok=True)
    reservoir: _PriorityReservoir | None = None
    first_metadata: dict[str, Any] | None = None
    state_resolution: dict[str, Any] | None = None
    action_resolution: dict[str, Any] | None = None
    state_grippers: set[int] = set()
    action_grippers: set[int] = set()
    episode_keys: set[str] = set()
    state_dim = action_dim = total_rows = episodes_scanned = 0
    requested_rows = max(s1cfg.max_samples, s3cfg.max_samples)
    capacity_rows = capacity_bytes = 0
    try:
        with partial_cache.open("wb") as cache_file:
            for episode in iter_rlds(
                args.dataset_dir, args.split, args.max_episodes, desc=f"Calibrating {args.split}",
            ):
                if episode.key in episode_keys:
                    raise ValueError(f"Duplicate episode key {episode.key!r} in split {args.split!r}")
                episode_keys.add(episode.key)
                if episode.state.ndim != 2 or episode.action.ndim != 2:
                    raise ValueError(
                        f"episode {episode.key}: state/action must be rank 2, got "
                        f"{episode.state.shape} and {episode.action.shape}"
                    )
                if len(episode.state) != len(episode.action):
                    raise ValueError(
                        f"episode {episode.key}: state/action frame counts differ: "
                        f"{len(episode.state)} != {len(episode.action)}"
                    )
                if reservoir is None:
                    first_metadata = episode.metadata
                    state_dim, action_dim = episode.state.shape[1], episode.action.shape[1]
                    state_grippers, state_resolution = resolve_gripper_indices(
                        args.state_gripper_indices, first_metadata, "state", state_dim,
                    )
                    action_grippers, action_resolution = resolve_gripper_indices(
                        args.action_gripper_indices, first_metadata, "action", action_dim,
                    )
                    print(
                        "Gripper resolution: "
                        f"state={sorted(state_grippers)} ({state_resolution['source']}), "
                        f"action={sorted(action_grippers)} ({action_resolution['source']})",
                        file=sys.stderr,
                    )
                    unresolved = [
                        signal for signal, resolution in (
                            ("state", state_resolution), ("action", action_resolution),
                        ) if resolution["status"] == "unresolved"
                    ]
                    if unresolved and not args.allow_unresolved_gripper:
                        options = " ".join(
                            f'--{signal}-gripper-indices ""' for signal in unresolved
                        )
                        raise ValueError(
                            "Could not resolve gripper dimensions for "
                            f"{', '.join(unresolved)}. Specify comma-separated indices, or "
                            f"explicitly confirm no gripper with: {options}. "
                            "Use --allow-unresolved-gripper only for exploratory runs."
                        )
                    for signal, resolution in (
                        ("state", state_resolution), ("action", action_resolution),
                    ):
                        if resolution["status"] in {"unresolved", "partial"}:
                            print(
                                f"WARNING: {signal} gripper resolution is "
                                f"{resolution['status']}; use --{signal}-gripper-indices "
                                "to specify it explicitly if needed.",
                                file=sys.stderr,
                            )
                    columns = 5 * (state_dim + action_dim)
                    bytes_per_row = columns * np.dtype(np.float32).itemsize + 8
                    budget = args.calibration_memory_mb * 1024 * 1024
                    capacity = min(requested_rows, budget // bytes_per_row)
                    reservoir = _PriorityReservoir(int(capacity), columns)
                    capacity_rows = int(capacity)
                    capacity_bytes = int(capacity * bytes_per_row)
                elif (episode.state.shape[1], episode.action.shape[1]) != (state_dim, action_dim):
                    raise ValueError(
                        f"episode {episode.key}: dimensions changed from "
                        f"({state_dim}, {action_dim}) to "
                        f"({episode.state.shape[1]}, {episode.action.shape[1]})"
                    )
                reservoir.update(_calibration_bundle(episode, s1cfg))
                pickle.dump(episode, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
                total_rows += len(episode.state)
                episodes_scanned += 1
        if reservoir is None or first_metadata is None:
            raise RuntimeError(f"No episodes found in split {args.split!r}")
        partial_cache.replace(cache_path)
    except BaseException:
        partial_cache.unlink(missing_ok=True)
        raise

    try:
        assert state_resolution is not None and action_resolution is not None
        s1_thresholds, s3 = _fit_sampled_thresholds(
            reservoir.sample, total_rows, state_dim, action_dim, s1cfg, s3cfg,
            state_grippers, action_grippers,
        )
        (output / "s1_thresholds.json").write_text(
            json.dumps(s1_thresholds, indent=2), encoding="utf-8",
        )
        (output / "s3_thresholds.json").write_text(
            json.dumps(s3, indent=2), encoding="utf-8",
        )
        retained_rows = len(reservoir.sample)
        del reservoir
    except BaseException:
        cache_path.unlink(missing_ok=True)
        raise

    counts = {"S1": 0, "S2": 0, "S3": 0, "flagged_any": 0}
    frames_path = output / "flagged_frames.jsonl"
    review_dir = output / "review"
    review_dir.mkdir(exist_ok=True)
    for stale_plot in review_dir.glob("episode_*.png"):
        stale_plot.unlink()
    plotted = 0
    try:
        with (output / "episodes.jsonl").open("w", encoding="utf-8") as episode_file, \
                frames_path.open("w", encoding="utf-8") as frame_file:
            for episode in _iter_episode_cache(cache_path, episodes_scanned):
                raw_action_is_delta = episode.metadata.get("action_is_delta")
                action_is_delta, action_semantics_source = resolve_action_is_delta(
                    raw_action_is_delta,
                )
                s1s = detect_s1(
                    episode.state, s1cfg, s1_thresholds["state"], state_grippers,
                )
                s1a = detect_s1(
                    episode.action, s1cfg, s1_thresholds["action"], action_grippers,
                )
                compatible, compatibility_reason = s2_is_compatible(
                    episode.metadata, episode.state.shape[1], episode.action.shape[1],
                )
                if compatible:
                    s2 = detect_s2(
                        episode.state, episode.action, s2cfg,
                        action_is_delta=action_is_delta,
                        ignored_dimensions=state_grippers | action_grippers,
                    )
                    s2["compatibility_reason"] = compatibility_reason
                else:
                    s2 = {
                        "flagged": False, "skipped": True,
                        "compatibility_reason": compatibility_reason, "dimensions": [],
                    }
                s2["raw_action_is_delta"] = raw_action_is_delta
                s2["action_semantics_source"] = action_semantics_source
                s3s = detect_s3(episode.state, s3["state"], state_grippers)
                s3a = detect_s3(episode.action, s3["action"], action_grippers)
                failed = []
                if s1s["flagged"] or s1a["flagged"]: failed.append("S1")
                if s2["flagged"]: failed.append("S2")
                if s3s["flagged"] or s3a["flagged"]: failed.append("S3")
                recommended_action = (
                    "drop_episode" if "S2" in failed else
                    "review_or_filter_frames" if failed else "keep"
                )
                report = {
                    "episode": episode.key, "num_frames": len(episode.state),
                    "failed_rules": failed, "review_required": bool(failed),
                    "recommended_action": recommended_action, "metadata": episode.metadata,
                    "s1_state": s1s, "s1_action": s1a, "s2": s2,
                    "s3_state": s3s, "s3_action": s3a,
                }
                for rule in failed: counts[rule] += 1
                counts["flagged_any"] += int(bool(failed))
                episode_file.write(json.dumps(report, ensure_ascii=False) + "\n")
                for rule, signal, result in (
                    ("S1", "state", s1s), ("S1", "action", s1a),
                    ("S3", "state", s3s), ("S3", "action", s3a),
                ):
                    for hit in result["hits"]:
                        frame_file.write(json.dumps({
                            "episode": episode.key, "rule": rule, "signal": signal, **hit,
                        }) + "\n")
                if failed and plotted < args.review_plots:
                    _plot_review(episode, report, review_dir / f"episode_{episode.key}.png")
                    plotted += 1
    finally:
        cache_path.unlink(missing_ok=True)

    summary = {
        "dataset_dir": str(args.dataset_dir), "split": args.split,
        "episodes_scanned": episodes_scanned, "frames_scanned": total_rows, "counts": counts,
        "state_dimensions": state_dim, "action_dimensions": action_dim,
        "state_gripper_indices": sorted(state_grippers),
        "action_gripper_indices": sorted(action_grippers),
        "gripper_resolution": {
            "state": state_resolution, "action": action_resolution,
        },
        "config": {"S1": asdict(s1cfg), "S2": asdict(s2cfg), "S3": asdict(s3cfg)},
        "calibration": {
            "sampling": "uniform_priority_reservoir",
            "requested_max_rows": requested_rows,
            "capacity_rows": capacity_rows,
            "retained_rows": retained_rows,
            "memory_budget_mb": args.calibration_memory_mb,
            "capacity_memory_bytes": capacity_bytes,
        },
        "reader": "selected_tfrecord_fields",
        "review_plots": plotted,
        "source_modified": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="seen_test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--state-gripper-indices", default="auto",
                        help="Comma-separated indices, empty, or 'auto' from RLDS metadata.")
    parser.add_argument("--action-gripper-indices", default="auto",
                        help="Comma-separated indices, empty, or 'auto' from RLDS metadata.")
    parser.add_argument(
        "--allow-unresolved-gripper", action="store_true",
        help="Continue when auto detection has no usable dimension metadata (unsafe).",
    )
    parser.add_argument("--s1-residual-z", type=float, default=8.0)
    parser.add_argument("--s1-acceleration-z", type=float, default=8.0)
    parser.add_argument("--s1-jerk-z", type=float, default=8.0)
    parser.add_argument("--s1-relative-scale-floor", type=float, default=0.002)
    parser.add_argument("--s1-max-samples", type=int, default=1_000_000)
    parser.add_argument("--s2-da-threshold", type=float, default=0.6)
    parser.add_argument("--s2-max-lag", type=int, default=10)
    parser.add_argument("--s2-min-active", type=int, default=10)
    parser.add_argument("--s2-allow-negative-lag", action="store_true")
    parser.add_argument("--s3-alpha", type=float, default=1.5)
    parser.add_argument("--s3-max-samples", type=int, default=1_000_000)
    parser.add_argument("--calibration-memory-mb", type=int, default=2048,
                        help="Hard cap for retained S1/S3 calibration rows (default: 2048 MiB).")
    parser.add_argument("--review-plots", type=int, default=12)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
