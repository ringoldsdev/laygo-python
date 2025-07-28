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
  distribution (get_handle) and resource management (shutdown).
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
