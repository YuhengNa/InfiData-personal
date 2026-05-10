"""VLM Backend interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """
        Run inference on a list of images.
        
        Args:
            images: List of images as numpy arrays (BGR format)
            prompt: The prompt text
            
        Returns:
            Dictionary with keys:
                - thought: str, reasoning process
                - transitions: List[int], frame indices where task switches occur
                - instructions: List[str], task descriptions for each segment
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass
    
    def warmup(self) -> None:
        """Optional warmup routine. Called before main loop."""
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup routine. Called on shutdown."""
        pass