"""
Defines the abstract base classes for context management in Laygo.

This module provides the core interfaces (IContextHandle and IContextManager)
that all context managers must implement, ensuring a consistent API for
state management across different execution environments (simple, threaded, parallel).
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Any


class IContextHandle(ABC):
  """
  An abstract base class for a picklable handle to a context manager.

  A handle contains the necessary information for a worker process to
  reconstruct a connection (a proxy) to the shared context.
  """

  @abstractmethod
  def create_proxy(self) -> "IContextManager":
    """
    Creates the appropriate context proxy instance from the handle's data.

    This method is called within a worker process to establish its own
    connection to the shared state.

    Returns:
        An instance of an IContextManager proxy.
    """
    raise NotImplementedError


class IContextManager(MutableMapping[str, Any], ABC):
  """
  Abstract base class for managing shared state (context) in a pipeline.

  This class defines the contract for all context managers, ensuring they
  provide a dictionary-like interface for state manipulation by inheriting
  from `collections.abc.MutableMapping`. It also includes methods for
  distribution (get_handle), resource management (shutdown), and context
  management (__enter__, __exit__).
  """

  @abstractmethod
  def get_handle(self) -> IContextHandle:
    """
    Returns a picklable handle for connecting from a worker process.

    This handle is serialized and sent to distributed workers, which then
    use it to create a proxy to the shared context.

    Returns:
        A picklable IContextHandle instance.
    """
    raise NotImplementedError

  @abstractmethod
  def shutdown(self) -> None:
    """
    Performs final synchronization and cleans up any resources.

    This method is responsible for releasing connections, shutting down
    background processes, or any other cleanup required by the manager.
    """
    raise NotImplementedError

  def __enter__(self) -> "IContextManager":
    """
    Enters the runtime context related to this object.

    Returns:
        The context manager instance itself.
    """
    return self

  def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
    """
    Exits the runtime context and performs cleanup.

    Args:
        exc_type: The exception type, if an exception was raised.
        exc_val: The exception instance, if an exception was raised.
        exc_tb: The traceback object, if an exception was raised.
    """
    self.shutdown()

  def to_dict(self) -> dict[str, Any]:
    """
    Returns a copy of the entire shared context as a standard
    Python dictionary.

    This operation is performed atomically using a lock to ensure a
    consistent snapshot of the context is returned.

    Returns:
        A standard dict containing a copy of the shared context.
    """
    # The dict() constructor iterates over the proxy and copies its items.
    # The lock ensures this happens atomically without race conditions.
    raise NotImplementedError
