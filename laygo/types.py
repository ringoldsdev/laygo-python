from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from typing import TypeVar

from laygo.context import IContextManager

In = TypeVar("In")
Out = TypeVar("Out")

type InternalTransformer[In, Out] = Callable[[list[In], IContextManager], list[Out]]


class BaseTransformer[In, Out](ABC):
  """
  Abstract base class for all transformer types.

  Defines the essential contract for a transformer, which is to be a callable
  that processes an iterable of data and yields an iterator of results,
  optionally using a context manager.
  """

  @abstractmethod
  def __call__(self, data: Iterable[In], context: IContextManager | None = None) -> Iterator[Out]:
    """
    Executes the transformation on a data source.
    This method must be implemented by all concrete transformer classes.
    """
    raise NotImplementedError


class ExecutionStrategy[In, Out](ABC):
  """Abstract base class for execution strategies.

  Strategies handle how transformer logic is executed (sequentially,
  threaded, in processes, etc.) but do not handle chunking.
  """

  @abstractmethod
  def execute(
    self,
    transformer_logic: InternalTransformer[In, Out],
    chunks: Iterator[list[In]],
    context: IContextManager,
  ) -> Iterator[list[Out]]:
    """Execute transformer logic on pre-chunked data.

    Args:
        transformer_logic: The transformation function to apply.
        chunks: Iterator of pre-chunked data.
        context: Context manager for the execution.

    Returns:
        Iterator of transformed chunks.
    """
    ...
