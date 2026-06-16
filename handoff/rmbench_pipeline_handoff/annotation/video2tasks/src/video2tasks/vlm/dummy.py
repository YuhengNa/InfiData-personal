"""Dummy VLM backend for testing."""

from typing import List, Dict, Any
import numpy as np
from .base import VLMBackend


class DummyBackend(VLMBackend):
    """Dummy backend that returns mock results without loading models."""
    
    @property
    def name(self) -> str:
        return "dummy"
    
    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """Return mock segmentation results."""
        n = len(images)
        
        # Simple heuristic: pretend to find a switch in the middle
        if n > 8:
            transitions = [n // 2]
            instructions = ["First task", "Second task"]
        else:
            transitions = []
            instructions = ["Single task"]
        
        return {
            "thought": f"Dummy analysis of {n} frames. No actual inference performed.",
            "transitions": transitions,
            "instructions": instructions
        }