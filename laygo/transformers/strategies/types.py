from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator

from laygo.context.types import IContextManager
from laygo.transformers.types import InternalTransformer


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
