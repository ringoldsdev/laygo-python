"""
A simple, dictionary-based context manager for single-process pipelines.
"""

from collections.abc import Iterator
from typing import Any

from laygo.context.types import IContextHandle
from laygo.context.types import IContextManager


class SimpleContextHandle(IContextHandle):
  """
  A handle for the SimpleContextManager that provides a reference back to the
  original manager instance.

  In a single-process environment, the "proxy" is the manager itself, ensuring
  all transformers in a chain share the exact same context dictionary.
  """

  def __init__(self, manager_instance: "IContextManager"):
    self._manager_instance = manager_instance

  def create_proxy(self) -> "IContextManager":
    """
    Returns the original SimpleContextManager instance.

    This ensures that in a non-distributed pipeline, all chained transformers
    operate on the same shared dictionary.
    """
    return self._manager_instance


class SimpleContextManager(IContextManager):
  """
  A basic context manager that uses a standard Python dictionary for state.

  This manager is suitable for single-threaded, single-process pipelines where
  no state needs to be shared across process boundaries. It is the default
  context manager for a Laygo pipeline.
  """

  def __init__(self, initial_context: dict[str, Any] | None = None) -> None:
    """
    Initializes the context manager with an optional dictionary.

    Args:
        initial_context: An optional dictionary to populate the context with.
    """
    self._context = dict(initial_context or {})

  def get_handle(self) -> IContextHandle:
    """
    Returns a handle that holds a reference back to this same instance.
    """
    return SimpleContextHandle(self)

  def __enter__(self) -> "SimpleContextManager":
    """
    Provides 'with' statement compatibility. No lock is needed for this
    simple, single-threaded context manager.
    """
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """
    Provides 'with' statement compatibility. No lock is needed for this
    simple, single-threaded context manager.
    """
    pass

  def __getitem__(self, key: str) -> Any:
    return self._context[key]

  def __setitem__(self, key: str, value: Any) -> None:
    self._context[key] = value

  def __delitem__(self, key: str) -> None:
    del self._context[key]

  def __iter__(self) -> Iterator[str]:
    return iter(self._context)

  def __len__(self) -> int:
    return len(self._context)

  def shutdown(self) -> None:
    """No-op for the simple context manager."""
    pass

  def to_dict(self) -> dict[str, Any]:
    """
    Returns a copy of the entire context as a standard Python dictionary.

    This operation is performed atomically to ensure consistency.
    """
    return self._context
