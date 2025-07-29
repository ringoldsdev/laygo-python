"""
Laygo Context Management Package.

This package provides different strategies for managing state (context)
within a data pipeline, from simple in-memory dictionaries to
process-safe managers for parallel execution.
"""

from .parallel import ParallelContextManager
from .simple import SimpleContextManager
from .types import IContextHandle
from .types import IContextManager

__all__ = [
  "IContextManager",
  "IContextHandle",
  "SimpleContextManager",
  "ParallelContextManager",
]
