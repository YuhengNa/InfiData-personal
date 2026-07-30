#!/usr/bin/env python3
"""Screen RLDS episodes with the Qwen S1/S2/S3 state-action checks.

The tool is deliberately non-destructive: it writes a review manifest and
plots, but never modifies the source RLDS/TFDS dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


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
    return np.abs(metric - center) / scale, center, scale


def _s1_metrics(values: np.ndarray, cfg: S1Config) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.asarray(values, dtype=np.float64)
    clean = _fill_nonfinite(raw)
    baseline = smooth(clean, cfg.median_kernel, cfg.savgol_window, cfg.savgol_polyorder)
    residual = np.abs(clean - baseline)
    acceleration = np.abs(np.diff(clean, n=2, axis=0, prepend=clean[[0]], append=clean[[-1]]))
    jerk = np.abs(np.diff(clean, n=3, axis=0, prepend=np.repeat(clean[[0]], 2, axis=0),
                          append=clean[[-1]]))
    jerk = jerk[: len(clean)]
    return baseline, {"residual": residual, "acceleration": acceleration, "jerk": jerk}


def fit_s1(episodes: Iterable[Episode], cfg: S1Config, signal: str) -> dict[str, Any]:
    values_chunks, metric_chunks = [], {"residual": [], "acceleration": [], "jerk": []}
    for episode in episodes:
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
    signal_span = np.nanquantile(values, 0.99, axis=0) - np.nanquantile(values, 0.01, axis=0)
    floor = np.maximum(cfg.scale_floor, cfg.relative_scale_floor * signal_span)
    result: dict[str, Any] = {"signal": signal, "sample_rows": len(values)}
    for name, metric in metrics.items():
        _, center, scale = _robust_z(metric, floor)
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
            computed[name] = {"z": z, "center": center, "scale": scale}
    else:
        computed = {}
        for name, metric in metrics.items():
            center = np.asarray(thresholds[name]["center"])
            scale = np.asarray(thresholds[name]["scale"])
            computed[name] = {
                "z": np.abs(metric - center) / scale,
                "center": center,
                "scale": scale,
            }
    residual, acceleration, jerk = (metrics[name] for name in ("residual", "acceleration", "jerk"))
    rz, az, jz = (computed[name]["z"] for name in ("residual", "acceleration", "jerk"))
    rscale, ascale, jscale = (computed[name]["scale"] for name in ("residual", "acceleration", "jerk"))
    mask = (rz >= cfg.residual_z) & ((az >= cfg.acceleration_z) | (jz >= cfg.jerk_z))
    mask |= ~np.isfinite(raw)
    ignored_dimensions = ignored_dimensions or set()
    for dim in ignored_dimensions:
        if 0 <= dim < mask.shape[1]:
            mask[:, dim] = False
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


def detect_s2(state: np.ndarray, action: np.ndarray, cfg: S2Config,
              action_is_delta: bool = False) -> dict[str, Any]:
    """Check trend alignment on dimensions with matching physical semantics."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    dims = min(state.shape[1], action.shape[1])
    if dims == 0 or len(state) < 4:
        return {"flagged": False, "reason": "insufficient_data", "dimensions": []}
    state = smooth(state[:, :dims])
    action = _fill_nonfinite(action[:, :dims])
    if action_is_delta:
        action = np.cumsum(action, axis=0)
    action = smooth(action)
    sd = np.diff(state, axis=0)
    ad = np.diff(action, axis=0)
    results = []
    for dim in range(dims):
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
            agreement = float(np.mean(np.sign(av) == np.sign(sv)))
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


def fit_s3(episodes: Iterable[Episode], cfg: S3Config,
           state_grippers: set[int], action_grippers: set[int]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    buckets: dict[str, list[np.ndarray]] = {"state": [], "action": []}
    totals = {"state": 0, "action": 0}
    for episode in episodes:
        for name, values in (("state", episode.state), ("action", episode.action)):
            buckets[name].append(np.asarray(values, dtype=np.float64))
            totals[name] += len(values)
            if totals[name] >= cfg.max_samples * 2:
                merged = np.concatenate(buckets[name], axis=0)
                keep = rng.choice(len(merged), cfg.max_samples, replace=False)
                buckets[name] = [merged[keep]]
                totals[name] = cfg.max_samples
    output: dict[str, Any] = {}
    for name, chunks in buckets.items():
        values = np.concatenate(chunks, axis=0)
        if len(values) > cfg.max_samples:
            values = values[rng.choice(len(values), cfg.max_samples, replace=False)]
        q01 = np.nanquantile(values, cfg.q_low, axis=0)
        q99 = np.nanquantile(values, cfg.q_high, axis=0)
        span = q99 - q01
        lower, upper = q01 - cfg.alpha * span, q99 + cfg.alpha * span
        grippers = state_grippers if name == "state" else action_grippers
        output[name] = {
            "q01": q01.tolist(), "q99": q99.tolist(),
            "lower": lower.tolist(), "upper": upper.tolist(),
            "gripper_indices": sorted(grippers), "sample_rows": int(len(values)),
        }
    output["config"] = asdict(cfg)
    return output


def detect_s3(values: np.ndarray, thresholds: dict[str, Any], grippers: set[int]) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float64)
    lower, upper = np.asarray(thresholds["lower"]), np.asarray(thresholds["upper"])
    mask = (raw < lower) | (raw > upper) | ~np.isfinite(raw)
    for dim in grippers:
        if 0 <= dim < mask.shape[1]:
            mask[:, dim] = False
    frames, dims = np.nonzero(mask)
    hits = [{
        "frame": frame, "dim": dim, "value": _json_number(raw[frame, dim]),
        "lower": float(lower[dim]), "upper": float(upper[dim]),
    } for frame, dim in zip(frames.tolist(), dims.tolist())]
    return {"flagged": bool(hits), "frames": sorted(set(frames.tolist())), "hits": hits}


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


def load_rlds(dataset_dir: Path, split: str, max_episodes: int | None) -> list[Episode]:
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    cameras = list(builder.info.features["steps"].feature["observation"]["images"].keys())
    decoders = {"steps": {"observation": {"images": {
        camera: tfds.decode.SkipDecoding() for camera in cameras
    }}}}
    dataset = builder.as_dataset(split=split, decoders=decoders)
    if max_episodes is not None:
        dataset = dataset.take(max_episodes)
    episodes = []
    for ordinal, raw in enumerate(dataset):
        metadata = _decode(raw["episode_metadata"])
        states, actions = [], []
        for step in raw["steps"].as_numpy_iterator():
            states.append(step["observation"]["state"])
            actions.append(step["action"])
        key = str(metadata.get("source_episode_index", metadata.get("episode_index", ordinal)))
        episodes.append(Episode(key, np.asarray(states), np.asarray(actions), metadata))
        print(f"\rLoaded {len(episodes)} episode(s)", end="", flush=True)
    print()
    return episodes


def _parse_indices(raw: str, metadata: dict[str, Any], layout_key: str) -> set[int]:
    if raw.strip().lower() != "auto":
        return {int(item) for item in raw.split(",") if item.strip()} if raw.strip() else set()
    layout = _state_action_schema(metadata).get(layout_key, [])
    return {index for index, name in enumerate(layout) if "gripper" in str(name).lower()}


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    episodes = load_rlds(args.dataset_dir, args.split, args.max_episodes)
    if not episodes:
        raise RuntimeError(f"No episodes found in split {args.split!r}")
    state_grippers = _parse_indices(
        args.state_gripper_indices, episodes[0].metadata, "state_layout",
    )
    action_grippers = _parse_indices(
        args.action_gripper_indices, episodes[0].metadata, "action_layout",
    )
    s1cfg = S1Config(residual_z=args.s1_residual_z, acceleration_z=args.s1_acceleration_z,
                     jerk_z=args.s1_jerk_z, relative_scale_floor=args.s1_relative_scale_floor)
    s2cfg = S2Config(da_threshold=args.s2_da_threshold, max_lag=args.s2_max_lag,
                     min_active=args.s2_min_active, flag_negative_lag=not args.s2_allow_negative_lag)
    s3cfg = S3Config(alpha=args.s3_alpha, max_samples=args.s3_max_samples)
    s1_thresholds = {
        "state": fit_s1(episodes, s1cfg, "state"),
        "action": fit_s1(episodes, s1cfg, "action"),
    }
    (output / "s1_thresholds.json").write_text(
        json.dumps(s1_thresholds, indent=2), encoding="utf-8",
    )
    s3 = fit_s3(episodes, s3cfg, state_grippers, action_grippers)
    (output / "s3_thresholds.json").write_text(json.dumps(s3, indent=2), encoding="utf-8")

    counts = {"S1": 0, "S2": 0, "S3": 0, "flagged_any": 0}
    reports = []
    frames_path = output / "flagged_frames.jsonl"
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as episode_file, \
            frames_path.open("w", encoding="utf-8") as frame_file:
        for episode in episodes:
            action_is_delta = bool(episode.metadata.get("action_is_delta", False))
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
                s2 = detect_s2(episode.state, episode.action, s2cfg, action_is_delta)
                s2["compatibility_reason"] = compatibility_reason
            else:
                s2 = {
                    "flagged": False, "skipped": True,
                    "compatibility_reason": compatibility_reason,
                    "dimensions": [],
                }
            s3s = detect_s3(episode.state, s3["state"], state_grippers)
            s3a = detect_s3(episode.action, s3["action"], action_grippers)
            failed = []
            if s1s["flagged"] or s1a["flagged"]: failed.append("S1")
            if s2["flagged"]: failed.append("S2")
            if s3s["flagged"] or s3a["flagged"]: failed.append("S3")
            if "S2" in failed:
                recommended_action = "drop_episode"
            elif failed:
                recommended_action = "review_or_filter_frames"
            else:
                recommended_action = "keep"
            report = {
                "episode": episode.key, "num_frames": len(episode.state),
                "failed_rules": failed,
                "review_required": bool(failed),
                "recommended_action": recommended_action,
                "metadata": episode.metadata,
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
                    frame_file.write(json.dumps({"episode": episode.key, "rule": rule,
                                                 "signal": signal, **hit}) + "\n")
            reports.append(report)

    review_dir = output / "review"
    review_dir.mkdir(exist_ok=True)
    for stale_plot in review_dir.glob("episode_*.png"):
        stale_plot.unlink()
    plotted = 0
    for episode, report in zip(episodes, reports):
        if report["failed_rules"] and plotted < args.review_plots:
            _plot_review(episode, report, review_dir / f"episode_{episode.key}.png")
            plotted += 1
    summary = {
        "dataset_dir": str(args.dataset_dir), "split": args.split,
        "episodes_scanned": len(episodes), "counts": counts,
        "state_dimensions": int(episodes[0].state.shape[1]),
        "action_dimensions": int(episodes[0].action.shape[1]),
        "state_gripper_indices": sorted(state_grippers),
        "action_gripper_indices": sorted(action_grippers),
        "config": {"S1": asdict(s1cfg), "S2": asdict(s2cfg), "S3": asdict(s3cfg)},
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
                        help="Comma-separated indices, empty, or 'auto' from state_layout.")
    parser.add_argument("--action-gripper-indices", default="auto",
                        help="Comma-separated indices, empty, or 'auto' from action_layout.")
    parser.add_argument("--s1-residual-z", type=float, default=8.0)
    parser.add_argument("--s1-acceleration-z", type=float, default=8.0)
    parser.add_argument("--s1-jerk-z", type=float, default=8.0)
    parser.add_argument("--s1-relative-scale-floor", type=float, default=0.002)
    parser.add_argument("--s2-da-threshold", type=float, default=0.6)
    parser.add_argument("--s2-max-lag", type=int, default=10)
    parser.add_argument("--s2-min-active", type=int, default=10)
    parser.add_argument("--s2-allow-negative-lag", action="store_true")
    parser.add_argument("--s3-alpha", type=float, default=1.5)
    parser.add_argument("--s3-max-samples", type=int, default=1_000_000)
    parser.add_argument("--review-plots", type=int, default=12)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
