from typing import Any, Dict, List, Optional

import base64
import json
import time

import cv2
import numpy as np
import requests

from .base import VLMBackend


def _encode_png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    t = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        print(f"[RemoteAPI] Failed to parse JSON directly: {e}")

    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(t[s : e + 1])
        except json.JSONDecodeError as e:
            print(f"[RemoteAPI] Failed to extract JSON from text: {e}")
            return {}
    return {}


class RemoteAPIBackend(VLMBackend):
    def __init__(
        self,
        url: str,
        api_key: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout_sec: float = 60.0,
    ):
        self.url = url
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout_sec = float(timeout_sec)

    @property
    def name(self) -> str:
        return "remote_api"

    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        png_hex = [_encode_png_b64(img) for img in images]
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "images_b64_png": png_hex,
        }

        headers = dict(self.headers)
        if self.api_key and "authorization" not in {k.lower(): v for k, v in headers.items()}:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t0 = time.time()
        r = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout_sec)
        latency_s = time.time() - t0

        if r.status_code != 200:
            print(
                f"[RemoteAPI] Error: status={r.status_code} latency_s={latency_s:.3f}"
            )
            return {}

        try:
            data = r.json()
        except json.JSONDecodeError as e:
            print(f"[RemoteAPI] Failed to parse response JSON: {e}")
            data = {}

        if isinstance(data, dict):
            if "transitions" in data or "instructions" in data:
                return data
            if "vlm_json" in data and isinstance(data["vlm_json"], dict):
                return data["vlm_json"]
            if "text" in data and isinstance(data["text"], str):
                parsed = _extract_json(data["text"])
                return parsed

        return {}
