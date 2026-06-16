"""Validation helpers for memory annotations."""

from typing import Any, Dict, List, Optional


def validate_memory_annotation(
    vlm_json: Dict[str, Any],
    n_images: int,
    nested: bool = False,
) -> Optional[str]:
    payload = vlm_json.get("memory") if nested else vlm_json.get("memory", vlm_json)
    if not isinstance(payload, dict):
        return "memory must be a JSON object"
    if n_images <= 0:
        return "memory annotation has no sampled images"
    if not isinstance(payload.get("thought"), str) or not payload["thought"].strip():
        return "memory.thought must be a non-empty string"

    transitions = payload.get("transitions")
    summaries = payload.get("summaries")
    event_types = payload.get("change_event_types")

    if not isinstance(transitions, list):
        return "memory.transitions must be a list"
    if any(isinstance(x, bool) or not isinstance(x, int) for x in transitions):
        return "memory.transitions must contain integers"
    if transitions != sorted(set(transitions)):
        return "memory.transitions must be sorted and unique"
    if any(x <= 0 or x >= n_images for x in transitions):
        return f"memory.transitions must be between 1 and {n_images - 1}"

    expected = len(transitions) + 1
    if not isinstance(summaries, list) or len(summaries) != expected:
        return f"memory.summaries must contain exactly {expected} items"
    if any(not isinstance(x, str) or not x.strip() for x in summaries):
        return "memory.summaries must contain non-empty strings"

    if not isinstance(event_types, list) or len(event_types) != expected:
        return f"memory.change_event_types must contain exactly {expected} items"
    if any(
        not isinstance(items, list)
        or not items
        or any(not isinstance(tag, str) or not tag.strip() for tag in items)
        for items in event_types
    ):
        return "each memory.change_event_types item must be a non-empty list of strings"

    return None


def validate_memory_coverage(records: List[Dict[str, Any]], nframes: int) -> Optional[str]:
    if nframes <= 0:
        return "video has no frames"
    if not records:
        return "no memory segments were produced"

    cursor = 0
    for rec in sorted(records, key=lambda x: int(x.get("start_frame", -1))):
        start = int(rec.get("start_frame", -1))
        end = int(rec.get("end_frame", -1))
        if not str(rec.get("summary", "")).strip():
            return f"memory segment {start}-{end} has an empty summary"
        if start != cursor:
            return f"expected next memory segment at frame {cursor}, got {start}"
        if end < start or end >= nframes:
            return f"invalid memory segment range {start}-{end} for {nframes} frames"
        cursor = end + 1

    if cursor != nframes:
        return f"memory coverage ends at frame {cursor - 1}, expected {nframes - 1}"
    return None
