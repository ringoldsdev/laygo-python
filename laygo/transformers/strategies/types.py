from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator

from laygo.context.types import IContextManager
from laygo.transformers.types import InternalTransformer

type ChunkGenerator[In] = Callable[[Iterable[In]], Iterator[list[In]]]


class ExecutionStrategy[In, Out](ABC):
  """Defines the contract for all execution strategies."""

  @abstractmethod
  def execute(
    self,
    transformer_logic: InternalTransformer[In, Out],
    chunk_generator: Callable[[Iterable[In]], Iterator[list[In]]],
    data: Iterable[In],
    context: IContextManager,
  ) -> Iterator[Out]:
    """Runs the transformation logic on the data."""
    raise NotImplementedError
