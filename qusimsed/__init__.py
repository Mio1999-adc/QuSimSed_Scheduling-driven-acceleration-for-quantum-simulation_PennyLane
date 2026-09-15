"""QuSim-Sed experiment framework.

This package deliberately separates scheduler evidence from performance
numbers.  A trace only calls itself CUDA-backed when the executor reports an
actual CUDA stream identifier.
"""

from .config import ExperimentConfig
from .core.cds import CDSRecord, RecordPool
from .core.scheduler import ResourceAwareScheduler

__all__ = ["ExperimentConfig", "CDSRecord", "RecordPool", "ResourceAwareScheduler"]
