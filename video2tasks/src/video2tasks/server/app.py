"""FastAPI server for job queue management."""

import os
import json
import time
import glob
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from ..config import Config, DatasetConfig
from .windowing import (
    read_video_info, build_windows, FrameExtractor,
    build_segments_via_cuts, build_memory_segments_via_cuts, Window
)


class SubmitModel(BaseModel):
    """Model for job result submission."""
    task_id: str
    vlm_output: str = ""
    vlm_json: Dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    meta: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class DatasetCtx:
    """Dataset context for processing."""
    data_root: str
    subset: str
    input_format: str
    video_key: str
    data_dir: str
    run_dir: str
    samples_dir: str
    sample_ids: List[str]
    sample_videos: Dict[str, str]
    sample_meta: Dict[str, Dict[str, Any]]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _replace_episode_records(path: Path, new_records: List[Dict[str, Any]], episode_index: int) -> None:
    existing = _load_jsonl(path) if path.exists() else []
    kept = [r for r in existing if int(r.get("episode_index", -1)) != int(episode_index)]
    merged = kept + new_records
    merged.sort(key=lambda r: (int(r.get("episode_index", -1)), int(r.get("segment_index", -1))))
    _write_jsonl(path, merged)


def _update_episode_parquet_subtasks(data_dir: Path, source_meta: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    parquet_rel = source_meta.get("parquet_path")
    if not parquet_rel:
        print(f"[Warn] No parquet_path found for episode {source_meta.get('episode_index')}; skip parquet update")
        return

    parquet_path = data_dir / str(parquet_rel)
    if not parquet_path.exists():
        print(f"[Warn] Parquet not found: {parquet_path}")
        return

    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "frame_index" not in df.columns:
        print(f"[Warn] frame_index column missing in {parquet_path}; skip parquet update")
        return

    for rec in records:
        mask = (df["frame_index"] >= int(rec["start_frame"])) & (df["frame_index"] <= int(rec["end_frame"]))
        df.loc[mask, "subtask"] = str(rec.get("subtask", ""))

    df.to_parquet(parquet_path, index=False)


def _parse_infidata_samples(data_dir: Path, video_key: str) -> tuple[List[str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    episodes_path = data_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"[Warn] InfiData episodes file not found: {episodes_path}")
        return [], {}, {}

    sample_ids = []
    sample_videos = {}
    sample_meta = {}

    for ep in _load_jsonl(episodes_path):
        episode_index = int(ep["episode_index"])
        sample_id = f"episode_{episode_index:06d}"
        video_paths = ep.get("video_paths") or {}
        video_rel = video_paths.get(video_key)

        if not video_rel and ep.get("parquet_path"):
            try:
                import pandas as pd

                parquet_path = data_dir / str(ep["parquet_path"])
                df = pd.read_parquet(parquet_path, columns=[f"video.{video_key}.path"])
                if not df.empty:
                    video_rel = df[f"video.{video_key}.path"].iloc[0]
            except Exception as exc:
                print(f"[Warn] Could not read video path from parquet for {sample_id}: {exc}")

        if not video_rel:
            print(f"[Warn] Missing video_paths.{video_key} for {sample_id}")
            continue

        video_path = (data_dir / video_rel).resolve()
        sample_ids.append(sample_id)
        sample_videos[sample_id] = str(video_path)
        sample_meta[sample_id] = {
            "episode_index": episode_index,
            "task": ep.get("task", ""),
            "source_dataset": ep.get("source_dataset", ""),
            "parquet_path": ep.get("parquet_path", ""),
            "fps": ep.get("fps", None),
            "video_key": video_key,
            "video_path": str(video_path),
        }

    return sample_ids, sample_videos, sample_meta


def parse_datasets(config: Config) -> List[DatasetCtx]:
    """Parse dataset configurations into contexts."""
    ctxs = []
    for ds in config.datasets:
        data_dir = Path(ds.root) / ds.subset
        run_dir = Path(config.run.base_dir) / ds.subset / config.run.run_id
        samples_dir = run_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        sample_videos: Dict[str, str] = {}
        sample_meta: Dict[str, Dict[str, Any]] = {}

        if ds.format == "infidata":
            sample_ids, sample_videos, sample_meta = _parse_infidata_samples(data_dir, ds.video_key)
        elif data_dir.exists():
            sample_ids = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
        else:
            sample_ids = []
        
        ctxs.append(DatasetCtx(
            data_root=ds.root,
            subset=ds.subset,
            input_format=ds.format,
            video_key=ds.video_key,
            data_dir=str(data_dir),
            run_dir=str(run_dir),
            samples_dir=str(samples_dir),
            sample_ids=sample_ids,
            sample_videos=sample_videos,
            sample_meta=sample_meta,
        ))
    return ctxs


def create_app(config: Config) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="Video2Tasks Server")
    
    # Initialize dataset contexts
    dataset_ctxs = parse_datasets(config)
    samples_dir_by_subset = {ctx.subset: ctx.samples_dir for ctx in dataset_ctxs}
    data_dir_by_subset = {ctx.subset: ctx.data_dir for ctx in dataset_ctxs}
    
    # Thread-safe job management
    queue_lock = threading.Lock()
    job_queue: List[Dict[str, Any]] = []
    inflight: Dict[str, Dict[str, Any]] = {}
    retry_counts: Dict[str, int] = {}
    
    # Per-sample locks
    _sample_locks: Dict[str, threading.Lock] = {}
    _sample_locks_lock = threading.Lock()
    
    def get_sample_lock(sample_key: str) -> threading.Lock:
        with _sample_locks_lock:
            if sample_key not in _sample_locks:
                _sample_locks[sample_key] = threading.Lock()
            return _sample_locks[sample_key]
    
    def sample_out_dir(samples_dir: str, sample_id: str) -> str:
        p = Path(samples_dir) / sample_id
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    
    def windows_jsonl_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "windows.jsonl")
    
    def segments_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "segments.json")

    def infidata_segments_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "segments_infidata.jsonl")

    def run_infidata_segments_path(run_dir: str) -> str:
        return str(Path(run_dir) / "segments_infidata.jsonl")

    def memory_segments_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / "memory_segments.jsonl")

    def run_memory_segments_path(run_dir: str) -> str:
        return str(Path(run_dir) / "memory_segments.jsonl")
    
    def done_marker_path(samples_dir: str, sample_id: str) -> str:
        return str(Path(sample_out_dir(samples_dir, sample_id)) / ".DONE")
    
    @app.get("/get_job")
    def get_job() -> Dict[str, Any]:
        with queue_lock:
            if not job_queue:
                return {"status": "empty"}
            job = job_queue.pop(0)
            inflight[job["task_id"]] = {"ts": time.time(), "job": job}
            return {"status": "ok", "data": job}
    
    @app.post("/submit_result")
    def submit_result(res: SubmitModel) -> Dict[str, str]:
        tid = res.task_id
        job_info = None
        
        with queue_lock:
            if tid in inflight:
                job_info = inflight.pop(tid)
        
        # Empty result: trigger retry
        if not res.vlm_json:
            if job_info:
                with queue_lock:
                    retry_counts[tid] = retry_counts.get(tid, 0) + 1
                    if retry_counts[tid] <= config.server.max_retries_per_job:
                        job_queue.insert(0, job_info["job"])
                        print(f"[Warn] Task {tid} empty, re-queueing (attempt {retry_counts[tid]})")
                    else:
                        print(f"[Err] Task {tid} failed max retries, dropping")
            return {"status": "retry_triggered"}
        
        subset = str(res.meta.get("subset", dataset_ctxs[0].subset if dataset_ctxs else "default"))
        sid = str(res.meta.get("sample_id", "unknown"))
        w_id = res.meta.get("window_id")
        
        samples_dir = samples_dir_by_subset.get(subset)
        if not samples_dir:
            samples_dir = str(Path(config.run.base_dir) / subset / config.run.run_id / "samples")
            Path(samples_dir).mkdir(parents=True, exist_ok=True)
        
        rec = {"task_id": tid, "window_id": w_id, "vlm_json": res.vlm_json}
        
        sample_key = f"{subset}::{sid}"
        with get_sample_lock(sample_key):
            with open(windows_jsonl_path(samples_dir, sid), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        return {"status": "received"}
    
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}
    
    # Producer loop
    def producer_loop():
        # Compute progress totals
        total = sum(len(ctx.sample_ids) for ctx in dataset_ctxs)
        progress_total = config.progress.total_override if config.progress.total_override > 0 else total
        
        done = 0
        for ctx in dataset_ctxs:
            for sid in ctx.sample_ids:
                if Path(done_marker_path(ctx.samples_dir, sid)).exists():
                    done += 1
        
        print(
            f"[Server] Started. IMG=PNG, "
            f"FIXED={config.windowing.target_width}x{config.windowing.target_height}, "
            f"FRAMES_PER_WINDOW={config.windowing.frames_per_window}\n"
            f"[Plan] DATASETS={[(c.data_dir, c.subset, c.input_format) for c in dataset_ctxs]}\n"
            f"[Plan] TARGETS={config.annotation.targets}, WRITE_BACK={config.infidata.write_back}, "
            f"UPDATE_PARQUET_SUBTASKS={config.infidata.update_parquet_subtasks}\n"
            f"[Resume] Already done: {done}/{progress_total} (computed_total={total})"
        )
        
        # Initialize states
        states = {}
        for ctx in dataset_ctxs:
            states[ctx.subset] = {
                "cur_idx": 0,
                "sample_status": {sid: 0 for sid in ctx.sample_ids},
            }
        
        dataset_idx = 0
        global_done = done
        
        while True:
            # Check inflight timeouts
            now = time.time()
            with queue_lock:
                expired = [
                    tid for tid, info in inflight.items()
                    if now - info["ts"] > config.server.inflight_timeout_sec
                ]
                for tid in expired:
                    job = inflight.pop(tid)["job"]
                    retry_counts[tid] = retry_counts.get(tid, 0) + 1
                    if retry_counts[tid] <= config.server.max_retries_per_job:
                        job_queue.append(job)
            
            # All datasets done
            if dataset_idx >= len(dataset_ctxs):
                if config.server.auto_exit_after_all_done:
                    print(f"[All Done] {global_done}/{progress_total}. Exiting.")
                    os._exit(0)
                time.sleep(1.0)
                continue
            
            ctx = dataset_ctxs[dataset_idx]
            st = states[ctx.subset]
            cur_idx = st["cur_idx"]
            sample_status = st["sample_status"]
            sample_ids = ctx.sample_ids
            
            # Current dataset done, wait for queue to clear
            if cur_idx >= len(sample_ids):
                with queue_lock:
                    if not job_queue and not inflight:
                        print(f"[Dataset] Completed {ctx.subset}. Switching to next...")
                        dataset_idx += 1
                time.sleep(0.2)
                continue
            
            # Produce jobs if queue not full
            with queue_lock:
                q_len = len(job_queue)
            
            if q_len < config.server.max_queue:
                sid = sample_ids[cur_idx]
                s_dir = Path(ctx.data_dir) / sid
                
                # Skip if already done
                if Path(done_marker_path(ctx.samples_dir, sid)).exists():
                    sample_status[sid] = 3
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue
                
                # Find video
                if ctx.input_format == "infidata":
                    mp4 = ctx.sample_videos.get(sid, "")
                else:
                    mp4s = list(s_dir.glob("Frame_*.mp4"))
                    mp4 = str(mp4s[0]) if mp4s else ""

                if not mp4 or not Path(mp4).exists():
                    print(f"[Warn] Missing video for {ctx.subset}/{sid}: {mp4}")
                    st["cur_idx"] += 1
                    time.sleep(0.01)
                    continue
                
                w_path = windows_jsonl_path(ctx.samples_dir, sid)
                
                # Step A: Generate window tasks
                if sample_status[sid] == 0:
                    try:
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        # Load completed windows
                        done_wids = set()
                        if Path(w_path).exists():
                            with open(w_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        done_wids.add(json.loads(line)["window_id"])
                                    except (json.JSONDecodeError, KeyError) as e:
                                        print(f"[Warn] Corrupted line in {w_path}: {e}")
                        
                        with FrameExtractor(mp4) as extractor:
                            cnt = 0
                            
                            for w in windows:
                                if w.window_id in done_wids:
                                    continue
                                
                                tid = f"{ctx.subset}::{sid}_w{w.window_id}"
                                
                                # Check if already active
                                active = False
                                with queue_lock:
                                    if any(j["task_id"] == tid for j in job_queue) or tid in inflight:
                                        active = True
                                
                                if active:
                                    continue
                                
                                job = {
                                    "task_id": tid,
                                    "images": extractor.get_many_b64(
                                        w.frame_ids,
                                        config.windowing.target_width,
                                        config.windowing.target_height,
                                        config.windowing.png_compression
                                    ),
                                    "meta": {
                                        "subset": ctx.subset,
                                        "sample_id": sid,
                                        "window_id": w.window_id,
                                        "frame_ids": w.frame_ids,
                                        **ctx.sample_meta.get(sid, {}),
                                    }
                                }
                                
                                with queue_lock:
                                    job_queue.append(job)
                                
                                cnt += 1
                                if cnt > 20:
                                    break
                        
                        if cnt == 0:
                            sample_status[sid] = 2
                    
                    except Exception as e:
                        print(f"[Err] {ctx.subset}/{sid}: {e}")
                        import traceback
                        traceback.print_exc()
                        st["cur_idx"] += 1
                
                # Step B: Finalize
                if sample_status[sid] == 2:
                    try:
                        fps, nframes = read_video_info(mp4)
                        windows = build_windows(
                            fps, nframes,
                            config.windowing.window_sec,
                            config.windowing.step_sec,
                            config.windowing.frames_per_window
                        )
                        
                        by_wid = {}
                        if Path(w_path).exists():
                            with open(w_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        d = json.loads(line)
                                        by_wid[d["window_id"]] = d
                                    except (json.JSONDecodeError, KeyError):
                                        pass
                        
                        if len(by_wid) >= len(windows):
                            print(f"[Finalize] {ctx.subset}/{sid}...")
                            
                            source_meta = ctx.sample_meta.get(sid, {})
                            episode_index = int(source_meta.get("episode_index", -1))
                            task = str(source_meta.get("task", ""))

                            if "subtask" in config.annotation.targets:
                                final_res = build_segments_via_cuts(
                                    sid, windows, by_wid, fps, nframes,
                                    config.windowing.frames_per_window
                                )
                                if source_meta:
                                    final_res["source_meta"] = source_meta

                                with open(segments_path(ctx.samples_dir, sid), "w", encoding="utf-8") as f:
                                    json.dump(final_res, f, indent=2, ensure_ascii=False)

                                if ctx.input_format == "infidata":
                                    infidata_records = []
                                    for seg in final_res.get("segments", []):
                                        start_frame = int(seg["start_frame"])
                                        end_frame = max(start_frame, int(seg["end_frame"]) - 1)
                                        infidata_records.append({
                                            "episode_index": episode_index,
                                            "segment_index": int(seg["seg_id"]),
                                            "start_frame": start_frame,
                                            "end_frame": end_frame,
                                            "task": task,
                                            "subtask": str(seg.get("instruction", "")),
                                            "annotation_status": "vlm_pseudo",
                                        })

                                    _write_jsonl(Path(infidata_segments_path(ctx.samples_dir, sid)), infidata_records)
                                    with open(run_infidata_segments_path(ctx.run_dir), "a", encoding="utf-8") as f:
                                        for rec in infidata_records:
                                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                                    if config.infidata.write_back:
                                        _replace_episode_records(
                                            Path(ctx.data_dir) / "meta" / "segments.jsonl",
                                            infidata_records,
                                            episode_index,
                                        )
                                        print(f"[WriteBack] Updated {ctx.subset}/meta/segments.jsonl for episode {episode_index}")

                                    if config.infidata.write_back and config.infidata.update_parquet_subtasks:
                                        _update_episode_parquet_subtasks(Path(ctx.data_dir), source_meta, infidata_records)
                                        print(f"[WriteBack] Updated parquet subtask column for episode {episode_index}")

                            if "memory" in config.annotation.targets:
                                memory_res = build_memory_segments_via_cuts(
                                    sid, windows, by_wid, fps, nframes,
                                    config.windowing.frames_per_window
                                )
                                if source_meta:
                                    memory_res["source_meta"] = source_meta

                                memory_records = []
                                for seg in memory_res.get("memory_segments", []):
                                    start_frame = int(seg["start_frame"])
                                    end_frame = max(start_frame, int(seg["end_frame"]) - 1)
                                    memory_records.append({
                                        "episode_index": episode_index,
                                        "segment_index": int(seg["seg_id"]),
                                        "start_frame": start_frame,
                                        "end_frame": end_frame,
                                        "start_timestamp": float(start_frame / fps),
                                        "end_timestamp": float(end_frame / fps),
                                        "task": task,
                                        "summary": str(seg.get("summary", "")),
                                        "change_event_type": seg.get("change_event_type", ["memory_updated"]),
                                        "evidence_start_frame": start_frame,
                                        "evidence_end_frame": end_frame,
                                        "annotation_status": "vlm_pseudo",
                                    })

                                _write_jsonl(Path(memory_segments_path(ctx.samples_dir, sid)), memory_records)
                                with open(run_memory_segments_path(ctx.run_dir), "a", encoding="utf-8") as f:
                                    for rec in memory_records:
                                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                                if config.infidata.write_back and ctx.input_format == "infidata":
                                    _replace_episode_records(
                                        Path(ctx.data_dir) / "meta" / "memory_segments.jsonl",
                                        memory_records,
                                        episode_index,
                                    )
                                    print(f"[WriteBack] Updated {ctx.subset}/meta/memory_segments.jsonl for episode {episode_index}")
                            
                            done_path = done_marker_path(ctx.samples_dir, sid)
                            already_done = Path(done_path).exists()
                            Path(done_path).touch()
                            
                            sample_status[sid] = 3
                            st["cur_idx"] += 1
                            
                            if not already_done:
                                global_done += 1
                            print(f"[Progress] {global_done}/{progress_total} (finished: {ctx.subset}/{sid})")
                    
                    except Exception as e:
                        print(f"[Err-Finalize] {ctx.subset}/{sid}: {e}")
            
            time.sleep(0.1)
    
    # Start producer thread
    producer_thread = threading.Thread(target=producer_loop, daemon=True)
    producer_thread.start()
    
    return app


def run_server(config: Config) -> None:
    """Run the server with given configuration."""
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower()
    )
