from collections.abc import Callable
import inspect
from typing import Any
from typing import TypeGuard


class PipelineContext(dict[str, Any]):
  """Generic, untyped context available to all pipeline operations.

  A dictionary-based context that can store arbitrary data shared across
  pipeline operations. This allows passing state and configuration between
  different stages of data processing.
  """

  pass


# Define the specific callables for clarity
ContextAwareCallable = Callable[[Any, PipelineContext], Any]
ContextAwareReduceCallable = Callable[[Any, Any, PipelineContext], Any]


def is_context_aware(func: Callable[..., Any]) -> TypeGuard[ContextAwareCallable]:
  """Check if a function is context-aware by inspecting its signature.

  A context-aware function accepts a PipelineContext as its second parameter,
  allowing it to access shared state during pipeline execution.

  Args:
      func: The function to inspect for context awareness.

  Returns:
      True if the function accepts more than one parameter (indicating it's
      context-aware), False otherwise.
  """
  return len(inspect.signature(func).parameters) > 1


def is_context_aware_reduce(func: Callable[..., Any]) -> TypeGuard[ContextAwareReduceCallable]:
  """Check if a reduce function is context-aware by inspecting its signature.

  A context-aware reduce function accepts an accumulator, current value,
  and PipelineContext as its three parameters.

  Args:
      func: The reduce function to inspect for context awareness.

  Returns:
      True if the function accepts more than two parameters (indicating it's
      context-aware), False otherwise.
  """
  return len(inspect.signature(func).parameters) > 2
