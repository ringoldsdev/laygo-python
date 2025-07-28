"""
A context manager for parallel and distributed processing using
multiprocessing.Manager to share state across processes.
"""

from collections.abc import Callable
from collections.abc import Iterator
import multiprocessing as mp
from multiprocessing.managers import DictProxy
from threading import Lock
from typing import Any
from typing import TypeVar

from laygo.context.types import IContextHandle
from laygo.context.types import IContextManager

R = TypeVar("R")


class ParallelContextHandle(IContextHandle):
  """
  A lightweight, picklable handle that carries the actual shared objects
  (the DictProxy and Lock) to worker processes.
  """

  def __init__(self, shared_dict: DictProxy, lock: Lock):
    self._shared_dict = shared_dict
    self._lock = lock

  def create_proxy(self) -> "IContextManager":
    """
    Creates a new ParallelContextManager instance that wraps the shared
    objects received by the worker process.
    """
    return ParallelContextManager(handle=self)


class ParallelContextManager(IContextManager):
  """
  A context manager that enables state sharing across processes.

  It operates in two modes:
  1. Main Mode: When created normally, it starts a multiprocessing.Manager
     and creates a shared dictionary and lock.
  2. Proxy Mode: When created from a handle, it wraps a DictProxy and Lock
     that were received from another process. It does not own the manager.
  """

  def __init__(self, initial_context: dict[str, Any] | None = None, handle: ParallelContextHandle | None = None):
    """
    Initializes the manager. If a handle is provided, it initializes in
    proxy mode; otherwise, it starts a new manager.
    """
    if handle:
      # --- PROXY MODE INITIALIZATION ---
      # This instance is a client wrapping objects from an existing server.
      self._manager = None  # Proxies do not own the manager process.
      self._shared_dict = handle._shared_dict
      self._lock = handle._lock
    else:
      # --- MAIN MODE INITIALIZATION ---
      # This instance owns the manager and its shared objects.
      self._manager = mp.Manager()
      self._shared_dict = self._manager.dict(initial_context or {})
      self._lock = self._manager.Lock()

    self._is_locked = False

  def _lock_context(self) -> None:
    """Acquire the lock for this context manager."""
    if not self._is_locked:
      self._lock.acquire()
      self._is_locked = True

  def _unlock_context(self) -> None:
    """Release the lock for this context manager."""
    if self._is_locked:
      self._lock.release()
      self._is_locked = False

  def _execute_locked(self, operation: Callable[[], R]) -> R:
    """A private helper to execute an operation within a lock."""
    if not self._is_locked:
      self._lock_context()
      try:
        return operation()
      finally:
        self._unlock_context()
    else:
      return operation()

  def get_handle(self) -> ParallelContextHandle:
    """
    Returns a picklable handle containing the shared dict and lock.
    Only the main instance can generate handles.
    """
    if not self._manager:
      raise TypeError("Cannot get a handle from a proxy context instance.")

    return ParallelContextHandle(self._shared_dict, self._lock)

  def shutdown(self) -> None:
    """
    Shuts down the background manager process.
    This is a no-op for proxy instances.
    """
    if self._manager:
      self._manager.shutdown()

  def __enter__(self) -> "ParallelContextManager":
    """Acquires the lock for use in a 'with' statement."""
    self._lock_context()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the lock."""
    self._unlock_context()

  def __getitem__(self, key: str) -> Any:
    return self._shared_dict[key]

  def __setitem__(self, key: str, value: Any) -> None:
    self._execute_locked(lambda: self._shared_dict.__setitem__(key, value))

  def __delitem__(self, key: str) -> None:
    self._execute_locked(lambda: self._shared_dict.__delitem__(key))

  def __iter__(self) -> Iterator[str]:
    # Iteration needs to copy the keys to be safe across processes
    return self._execute_locked(lambda: iter(list(self._shared_dict.keys())))

  def __len__(self) -> int:
    return self._execute_locked(lambda: len(self._shared_dict))
