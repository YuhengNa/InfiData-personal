"""Video2Tasks - Multi-task video to single-task segments for VLA training."""

__version__ = "0.1.0"

from .config import Config, DatasetConfig, RunConfig, ServerConfig, WorkerConfig

__all__ = [
    "Config",
    "DatasetConfig", 
    "RunConfig",
    "ServerConfig",
    "WorkerConfig",
]