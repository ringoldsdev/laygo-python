# pipeline.py

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
import itertools
import multiprocessing as mp
from typing import Any
from typing import TypeVar
from typing import overload

from laygo.helpers import PipelineContext
from laygo.helpers import is_context_aware
from laygo.transformers.transformer import Transformer

T = TypeVar("T")
PipelineFunction = Callable[[T], Any]


class Pipeline[T]:
  """
  Manages a data source and applies transformers to it.
  Always uses a multiprocessing-safe shared context.
  """

  def __init__(self, *data: Iterable[T]):
    if len(data) == 0:
      raise ValueError("At least one data source must be provided to Pipeline.")
    self.data_source: Iterable[T] = itertools.chain.from_iterable(data) if len(data) > 1 else data[0]
    self.processed_data: Iterator = iter(self.data_source)

    # Always create a shared context with multiprocessing manager
    self._manager = mp.Manager()
    self.ctx = self._manager.dict()
    # Add a shared lock to the context for safe concurrent updates
    self.ctx["lock"] = self._manager.Lock()

    # Store reference to original context for final synchronization
    self._original_context_ref: PipelineContext | None = None

  def __del__(self):
    """Clean up the multiprocessing manager when the pipeline is destroyed."""
    try:
      self._sync_context_back()
      self._manager.shutdown()
    except Exception:
      pass  # Ignore errors during cleanup

  def context(self, ctx: PipelineContext) -> "Pipeline[T]":
    """
    Updates the pipeline context and stores a reference to the original context.
    When the pipeline finishes processing, the original context will be updated
    with the final pipeline context data.
    """
    # Store reference to the original context
    self._original_context_ref = ctx
    # Copy the context data to the pipeline's shared context
    self.ctx.update(ctx)
    return self

  def _sync_context_back(self) -> None:
    """
    Synchronize the final pipeline context back to the original context reference.
    This is called after processing is complete.
    """
    if self._original_context_ref is not None:
      # Copy the final context state back to the original context reference
      final_context_state = dict(self.ctx)
      final_context_state.pop("lock", None)  # Remove non-serializable lock
      self._original_context_ref.clear()
      self._original_context_ref.update(final_context_state)

  def transform[U](self, t: Callable[[Transformer[T, T]], Transformer[T, U]]) -> "Pipeline[U]":
    """
    Shorthand method to apply a transformation using a lambda function.
    Creates a Transformer under the hood and applies it to the pipeline.

    Args:
        t: A callable that takes a transformer and returns a transformed transformer

    Returns:
        A new Pipeline with the transformed data
    """
    # Create a new transformer and apply the transformation function
    transformer = t(Transformer[T, T]())
    return self.apply(transformer)

  @overload
  def apply[U](self, transformer: Transformer[T, U]) -> "Pipeline[U]": ...

  @overload
  def apply[U](self, transformer: Callable[[Iterable[T]], Iterator[U]]) -> "Pipeline[U]": ...

  @overload
  def apply[U](self, transformer: Callable[[Iterable[T], PipelineContext], Iterator[U]]) -> "Pipeline[U]": ...

  def apply[U](
    self,
    transformer: Transformer[T, U]
    | Callable[[Iterable[T]], Iterator[U]]
    | Callable[[Iterable[T], PipelineContext], Iterator[U]],
  ) -> "Pipeline[U]":
    """
    Applies a transformer to the current data source. The pipeline's
    managed context is passed down.
    """
    match transformer:
      case Transformer():
        # The transformer is called with self.ctx, which is the
        # shared mp.Manager.dict proxy when inside a 'with' block.
        self.processed_data = transformer(self.processed_data, self.ctx)  # type: ignore
      case _ if callable(transformer):
        if is_context_aware(transformer):
          processed_transformer = transformer
        else:
          processed_transformer = lambda data, ctx: transformer(data)  # type: ignore  # noqa: E731
        self.processed_data = processed_transformer(self.processed_data, self.ctx)  # type: ignore
      case _:
        raise TypeError("Transformer must be a Transformer instance or a callable function")

    return self  # type: ignore

  # ... The rest of the Pipeline class (transform, __iter__, to_list, etc.) remains unchanged ...
  def __iter__(self) -> Iterator[T]:
    """Allows the pipeline to be iterated over."""
    yield from self.processed_data

  def to_list(self) -> list[T]:
    """Executes the pipeline and returns the results as a list."""
    return list(self.processed_data)

  def each(self, function: PipelineFunction[T]) -> None:
    """Applies a function to each element (terminal operation)."""
    # Context needs to be accessed from the function if it's context-aware,
    # but the pipeline itself doesn't own a context. This is a design choice.
    # For simplicity, we assume the function is not context-aware here
    # or that context is handled within the Transformers.
    for item in self.processed_data:
      function(item)

  def first(self, n: int = 1) -> list[T]:
    """Gets the first n elements of the pipeline (terminal operation)."""
    assert n >= 1, "n must be at least 1"
    return list(itertools.islice(self.processed_data, n))

  def consume(self) -> None:
    """Consumes the pipeline without returning results."""
    for _ in self.processed_data:
      pass
