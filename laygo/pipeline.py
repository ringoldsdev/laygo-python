# pipeline.py
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import itertools
import multiprocessing as mp
from queue import Queue
from typing import Any
from typing import TypeVar
from typing import overload

from laygo.helpers import PipelineContext
from laygo.helpers import is_context_aware
from laygo.transformers.transformer import Transformer
from laygo.transformers.transformer import passthrough_chunks

T = TypeVar("T")
U = TypeVar("U")
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

  def branch(
    self,
    branches: dict[str, Transformer[T, Any]],
    batch_size: int = 1000,
    max_batch_buffer: int = 1,
    use_queue_chunks: bool = True,
  ) -> dict[str, list[Any]]:
    """Forks the pipeline into multiple branches for concurrent, parallel processing."""
    if not branches:
      self.consume()
      return {}

    source_iterator = self.processed_data
    branch_items = list(branches.items())
    num_branches = len(branch_items)
    final_results: dict[str, list[Any]] = {}

    queues = [Queue(maxsize=max_batch_buffer) for _ in range(num_branches)]

    def producer() -> None:
      """Reads from the source and distributes batches to ALL branch queues."""
      # Use itertools.batched for clean and efficient batch creation.
      for batch_tuple in itertools.batched(source_iterator, batch_size):
        # The batch is a tuple; convert to a list for consumers.
        batch_list = list(batch_tuple)
        for q in queues:
          q.put(batch_list)

      # Signal to all consumers that the stream is finished.
      for q in queues:
        q.put(None)

    def consumer(transformer: Transformer, queue: Queue) -> list[Any]:
      """Consumes batches from a queue and runs them through a transformer."""

      def stream_from_queue() -> Iterator[T]:
        while (batch := queue.get()) is not None:
          yield batch

      if use_queue_chunks:
        transformer = transformer.set_chunker(passthrough_chunks)

      result_iterator = transformer(stream_from_queue(), self.ctx)  # type: ignore
      return list(result_iterator)

    with ThreadPoolExecutor(max_workers=num_branches + 1) as executor:
      executor.submit(producer)

      future_to_name = {
        executor.submit(consumer, transformer, queues[i]): name for i, (name, transformer) in enumerate(branch_items)
      }

      for future in as_completed(future_to_name):
        name = future_to_name[future]
        try:
          final_results[name] = future.result()
        except Exception as e:
          print(f"Branch '{name}' raised an exception: {e}")
          final_results[name] = []

    return final_results

  def buffer(self, size: int, batch_size: int = 1000) -> "Pipeline[T]":
    """Inserts a buffer in the pipeline to allow downstream processing to read ahead.

    This creates a background thread that reads from the upstream data source
    and fills a queue, decoupling the upstream and downstream stages.

    Args:
        size: The number of **batches** to hold in the buffer.
        batch_size: The number of items to accumulate per batch.

    Returns:
        The pipeline instance for method chaining.
    """
    source_iterator = self.processed_data

    def _buffered_stream() -> Iterator[T]:
      queue = Queue(maxsize=size)
      # We only need one background thread for the producer.
      executor = ThreadPoolExecutor(max_workers=1)

      def _producer() -> None:
        """The producer reads from the source and fills the queue."""
        try:
          for batch_tuple in itertools.batched(source_iterator, batch_size):
            queue.put(list(batch_tuple))
        finally:
          # Always put the sentinel value to signal the end of the stream.
          queue.put(None)

      # Start the producer in the background thread.
      executor.submit(_producer)

      try:
        # The main thread becomes the consumer.
        while (batch := queue.get()) is not None:
          yield from batch
      finally:
        # Ensure the background thread is cleaned up.
        executor.shutdown(wait=False, cancel_futures=True)

    self.processed_data = _buffered_stream()
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
