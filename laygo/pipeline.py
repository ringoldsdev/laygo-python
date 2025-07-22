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
from laygo.transformers.threaded import ThreadedTransformer
from laygo.transformers.transformer import Transformer

T = TypeVar("T")
PipelineFunction = Callable[[T], Any]


class Pipeline[T]:
  """Manages a data source and applies transformers to it.

  A Pipeline provides a high-level interface for data processing by chaining
  transformers together. It automatically manages a multiprocessing-safe
  shared context that can be accessed by all transformers in the chain.
  """

  def __init__(self, *data: Iterable[T]) -> None:
    """Initialize a pipeline with one or more data sources.

    Args:
        *data: One or more iterable data sources. If multiple sources are
               provided, they will be chained together.

    Raises:
        ValueError: If no data sources are provided.
    """
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

  def __del__(self) -> None:
    """Clean up the multiprocessing manager when the pipeline is destroyed."""
    try:
      self._sync_context_back()
      self._manager.shutdown()
    except Exception:
      pass

  def context(self, ctx: PipelineContext) -> "Pipeline[T]":
    """Update the pipeline context and store a reference to the original context.

    When the pipeline finishes processing, the original context will be updated
    with the final pipeline context data.

    Args:
        ctx: The pipeline context to use for this pipeline execution.

    Returns:
        The pipeline instance for method chaining.
    """
    # Store reference to the original context
    self._original_context_ref = ctx
    # Copy the context data to the pipeline's shared context
    self.ctx.update(ctx)
    return self

  def _sync_context_back(self) -> None:
    """Synchronize the final pipeline context back to the original context reference.

    This is called after processing is complete to update the original
    context with any changes made during pipeline execution.
    """
    if self._original_context_ref is not None:
      # Copy the final context state back to the original context reference
      final_context_state = dict(self.ctx)
      final_context_state.pop("lock", None)  # Remove non-serializable lock
      self._original_context_ref.clear()
      self._original_context_ref.update(final_context_state)

  def transform[U](self, t: Callable[[Transformer[T, T]], Transformer[T, U]]) -> "Pipeline[U]":
    """Apply a transformation using a lambda function.

    Creates a Transformer under the hood and applies it to the pipeline.
    This is a shorthand method for simple transformations.

    Args:
        t: A callable that takes a transformer and returns a transformed transformer.

    Returns:
        A new Pipeline with the transformed data.
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
    """Apply a transformer to the current data source.

    The pipeline's managed context is passed down to the transformer.

    Args:
        transformer: Either a Transformer instance or a callable function
                    that processes the data.

    Returns:
        A new Pipeline with the transformed data.

    Raises:
        TypeError: If the transformer is not a supported type.
    """
    match transformer:
      case Transformer():
        self.processed_data = transformer(self.processed_data, self.ctx)  # type: ignore
      case _ if callable(transformer):
        if is_context_aware(transformer):
          self.processed_data = transformer(self.processed_data, self.ctx)  # type: ignore
        else:
          self.processed_data = transformer(self.processed_data)  # type: ignore
      case _:
        raise TypeError("Transformer must be a Transformer instance or a callable function")

    return self  # type: ignore

  def branch(self, branches: dict[str, Transformer[T, Any]]) -> dict[str, list[Any]]:
    """Forks the pipeline, sending all data to multiple branches and returning the last chunk.

    This is a **terminal operation** that implements a fan-out pattern.
    It consumes the pipeline's data, sends the **entire dataset** to each
    branch transformer, and continuously **overwrites** a shared context value
    with the latest processed chunk. The final result is a dictionary
    containing only the **last processed chunk** for each branch.

    Args:
        branches: A dictionary where keys are branch names (str) and values
                  are `Transformer` instances.

    Returns:
        A dictionary where keys are the branch names and values are lists
        of items from the last processed chunk for that branch.
    """
    if not branches:
      self.consume()
      return {}

    # 1. Build a single "fan-out" transformer by chaining taps.
    fan_out_transformer = Transformer[T, T]()

    for name, branch_transformer in branches.items():
      # Create a "collector" that runs the user's logic and then
      # overwrites the context with its latest chunk.
      collector = Transformer.from_transformer(branch_transformer)

      # This is the side-effect operation that overwrites the context.
      def overwrite_context_with_chunk(chunk: list[Any], ctx: PipelineContext, name=name) -> list[Any]:
        # This is an atomic assignment for manager dicts; no lock needed.
        ctx[name] = chunk
        # Return the chunk unmodified to satisfy the _pipe interface.
        return chunk

      # Add this as the final step in the collector's pipeline.
      collector._pipe(overwrite_context_with_chunk)

      # Tap the main transformer. The collector will run as a side-effect.
      fan_out_transformer.tap(collector)

    # 2. Apply the fan-out transformer and consume the entire pipeline.
    self.apply(fan_out_transformer).consume()

    # 3. Collect the final state from the context.
    final_results = {name: self.ctx.get(name, []) for name in branches}

    return final_results

  def buffer(self, size: int) -> "Pipeline[T]":
    """Buffer the pipeline using threaded processing.

    Args:
        size: The number of worker threads to use for buffering.

    Returns:
        The pipeline instance for method chaining.
    """
    self.apply(ThreadedTransformer(max_workers=size))
    return self

  def __iter__(self) -> Iterator[T]:
    """Allow the pipeline to be iterated over.

    Returns:
        An iterator over the processed data.
    """
    yield from self.processed_data

  def to_list(self) -> list[T]:
    """Execute the pipeline and return the results as a list.

    Returns:
        A list containing all processed items from the pipeline.
    """
    return list(self.processed_data)

  def each(self, function: PipelineFunction[T]) -> None:
    """Apply a function to each element (terminal operation).

    Args:
        function: The function to apply to each element.
    """
    for item in self.processed_data:
      function(item)

  def first(self, n: int = 1) -> list[T]:
    """Get the first n elements of the pipeline (terminal operation).

    Args:
        n: The number of elements to retrieve.

    Returns:
        A list containing the first n elements.

    Raises:
        AssertionError: If n is less than 1.
    """
    assert n >= 1, "n must be at least 1"
    return list(itertools.islice(self.processed_data, n))

  def consume(self) -> None:
    """Consume the pipeline without returning results.

    This is useful when you want to execute the pipeline for side effects
    without collecting the results.
    """
    for _ in self.processed_data:
      pass
