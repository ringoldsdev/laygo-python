"""
Laygo - A lightweight Python library for building resilient, in-memory data pipelines
"""

__version__ = "VERSION_NUMBER"

from .errors import ErrorHandler
from .helpers import PipelineContext
from .pipeline import Pipeline
from .transformers.parallel import ParallelTransformer
from .transformers.transformer import Transformer

__all__ = [
  "Pipeline",
  "Transformer",
  "ParallelTransformer",
  "PipelineContext",
  "ErrorHandler",
]
