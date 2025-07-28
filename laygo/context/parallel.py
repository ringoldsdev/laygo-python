"""
A context manager for parallel and distributed processing using
multiprocessing.Manager to share state across processes.
"""

from collections.abc import Iterator
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Lock
from typing import Any

from laygo.context.types import IContextHandle
from laygo.context.types import IContextManager


class _ParallelStateManager(BaseManager):
  """A custom manager to expose a shared dictionary and lock."""

  pass


class ParallelContextHandle(IContextHandle):
  """
  A lightweight, picklable "blueprint" for recreating a connection to the
  shared context in a different process.
  """

  def __init__(self, address: tuple[str, int], manager_class: type["ParallelContextManager"]):
    self.address = address
    self.manager_class = manager_class

  def create_proxy(self) -> "IContextManager":
    """
    Creates a new instance of the ParallelContextManager in "proxy" mode
    by initializing it with this handle.
    """
    return self.manager_class(handle=self)


class ParallelContextManager(IContextManager):
  """
  A context manager that uses a background multiprocessing.Manager to enable
  state sharing across different processes.

  This single class operates in two modes:
  1. Server Mode (when created normally): It starts and manages the background
     server process that holds the shared state.
  2. Proxy Mode (when created with a handle): It acts as a client, connecting
     to an existing server process to manipulate the shared state.
  """

  def __init__(self, initial_context: dict[str, Any] | None = None, handle: ParallelContextHandle | None = None):
    """
    Initializes the manager. If a handle is provided, it initializes in
    proxy mode; otherwise, it starts a new server.
    """
    if handle:
      # --- PROXY MODE INITIALIZATION ---
      # This instance is a client connecting to an existing server.
      self._is_proxy = True
      self._manager_server = None  # Proxies do not own the server process.

      manager = _ParallelStateManager(address=handle.address)
      manager.connect()
      self._manager = manager

    else:
      # --- SERVER MODE INITIALIZATION ---
      # This is the main instance that owns the server process.
      self._is_proxy = False
      manager = mp.Manager()  # type: ignore
      _ParallelStateManager.register("get_dict", callable=lambda: manager.dict(initial_context or {}))
      _ParallelStateManager.register("get_lock", callable=lambda: manager.Lock())

      self._manager_server = _ParallelStateManager(address=("", 0))
      self._manager_server.start()
      self._manager = self._manager_server

    # Common setup for both modes
    self._shared_dict: DictProxy = self._manager.get_dict()  # type: ignore
    self._lock: Lock = self._manager.get_lock()  # type: ignore

  def get_handle(self) -> ParallelContextHandle:
    """
    Returns a picklable handle for reconstruction in a worker.
    Only the main server instance can generate handles.
    """
    if self._is_proxy or not self._manager_server:
      raise TypeError("Cannot get a handle from a proxy context instance.")

    return ParallelContextHandle(
      address=self._manager_server.address,  # type: ignore
      manager_class=self.__class__,  # Pass its own class for reconstruction
    )

  def shutdown(self) -> None:
    """
    Shuts down the background manager process.
    This is a no-op for proxy instances, as only the main instance
    should control the server's lifecycle.
    """
    if not self._is_proxy and self._manager_server:
      self._manager_server.shutdown()

  def __enter__(self) -> "ParallelContextManager":
    """Acquires the lock for use in a 'with' statement."""
    self._lock.acquire()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the lock."""
    self._lock.release()

  def __getitem__(self, key: str) -> Any:
    with self._lock:
      return self._shared_dict[key]

  def __setitem__(self, key: str, value: Any) -> None:
    with self._lock:
      self._shared_dict[key] = value

  def __delitem__(self, key: str) -> None:
    with self._lock:
      del self._shared_dict[key]

  def __iter__(self) -> Iterator[str]:
    with self._lock:
      return iter(list(self._shared_dict.keys()))

  def __len__(self) -> int:
    with self._lock:
      return len(self._shared_dict)
