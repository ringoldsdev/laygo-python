# pipeline.py
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import itertools
from queue import Queue
from typing import Any
from typing import TypeVar
from typing import overload

from laygo.context import IContextManager
from laygo.context.parallel import ParallelContextManager
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

  The Pipeline supports both streaming and batch processing patterns, with
  built-in support for buffering, branching (fan-out), and parallel processing.

  Example:
      >>> data = [1, 2, 3, 4, 5]
      >>> result = (Pipeline(data)
      ...           .transform(lambda t: t.filter(lambda x: x % 2 == 0))
      ...           .transform(lambda t: t.map(lambda x: x * 2))
      ...           .to_list())
      >>> result  # [4, 8]

  Note:
      Most pipeline operations consume the internal iterator, making the
      pipeline effectively single-use unless the data source is re-initialized.
  """

  def __init__(self, *data: Iterable[T], context_manager: IContextManager | None = None) -> None:
    """Initialize a pipeline with one or more data sources.

    Args:
        *data: One or more iterable data sources. If multiple sources are
               provided, they will be chained together.
        context_manager: An instance of a class that implements IContextManager.
                         If None, a ParallelContextManager is used by default.

    Raises:
        ValueError: If no data sources are provided.
    """
    if len(data) == 0:
      raise ValueError("At least one data source must be provided to Pipeline.")
    self.data_source: Iterable[T] = itertools.chain.from_iterable(data) if len(data) > 1 else data[0]
    self.processed_data: Iterator = iter(self.data_source)

    # Rule 1: Pipeline creates a simple context manager by default.
    self.context_manager = context_manager or ParallelContextManager()

  def __del__(self) -> None:
    """Clean up the context manager when the pipeline is destroyed."""
    if hasattr(self, "context_manager"):
      self.context_manager.shutdown()

  def context(self, ctx: dict[str, Any]) -> "Pipeline[T]":
    """Update the pipeline's context manager with values from a dictionary.

    The provided context will be used during pipeline execution and any
    modifications made by transformers will be synchronized back to the
    original context when the pipeline finishes processing.

    Args:
        ctx: The pipeline context dictionary to use for this pipeline execution.
             This should be a mutable dictionary-like object that transformers
             can use to share state and communicate.

    Returns:
        The pipeline instance for method chaining.

    Note:
        Changes made to the context during pipeline execution will be
        automatically synchronized back to the original context object
        when the pipeline is destroyed or processing completes.
    """
    self.context_manager.update(ctx)
    return self

  def _sync_context_back(self) -> None:
    """Synchronize the final pipeline context back to the original context reference.

    This is called after processing is complete to update the original
    context with any changes made during pipeline execution.
    """
    # This method is kept for backward compatibility but is no longer needed
    # since we use the context manager directly
    pass

  def transform[U](self, t: Callable[[Transformer[T, T]], Transformer[T, U]]) -> "Pipeline[U]":
    """Apply a transformation using a lambda function.

    Creates a Transformer under the hood and applies it to the pipeline.
    This is a shorthand method for simple transformations that allows
    chaining transformer operations in a functional style.

    Args:
        t: A callable that takes a transformer and returns a transformed transformer.
           Typically used with lambda expressions like:
           `lambda t: t.map(func).filter(predicate)`

    Returns:
        A new Pipeline with the transformed data type.

    Example:
        >>> pipeline = Pipeline([1, 2, 3, 4, 5])
        >>> result = pipeline.transform(lambda t: t.filter(lambda x: x % 2 == 0).map(lambda x: x * 2))
        >>> result.to_list()  # [4, 8]
    """
    # Create a new transformer and apply the transformation function
    transformer = t(Transformer[T, T]())
    return self.apply(transformer)

  @overload
  def apply[U](self, transformer: Transformer[T, U]) -> "Pipeline[U]": ...

  @overload
  def apply[U](self, transformer: Callable[[Iterable[T]], Iterator[U]]) -> "Pipeline[U]": ...

  @overload
  def apply[U](self, transformer: Callable[[Iterable[T], IContextManager], Iterator[U]]) -> "Pipeline[U]": ...

  def apply[U](
    self,
    transformer: Transformer[T, U]
    | Callable[[Iterable[T]], Iterator[U]]
    | Callable[[Iterable[T], IContextManager], Iterator[U]],
  ) -> "Pipeline[U]":
    """Apply a transformer to the current data source.

    This method accepts various types of transformers and applies them to
    the pipeline data. The pipeline's managed context is automatically
    passed to context-aware transformers.

    Args:
        transformer: One of the following:
                    - A Transformer instance (preferred for complex operations)
                    - A callable function that takes an iterable and returns an iterator
                    - A context-aware callable that takes an iterable and context

    Returns:
        The same Pipeline instance with transformed data (for method chaining).

    Raises:
        TypeError: If the transformer is not a supported type.

    Example:
        >>> pipeline = Pipeline([1, 2, 3])
        >>> # Using a Transformer instance
        >>> pipeline.apply(createTransformer(int).map(lambda x: x * 2))
        >>> # Using a simple function
        >>> pipeline.apply(lambda data: (x * 2 for x in data))
    """
    match transformer:
      case Transformer():
        # Pass the pipeline's context manager to the transformer
        self.processed_data = transformer(self.processed_data, self.context_manager)  # type: ignore
      case _ if callable(transformer):
        if is_context_aware(transformer):
          self.processed_data = transformer(self.processed_data, self.context_manager)  # type: ignore
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
    """Forks the pipeline into multiple branches for concurrent, parallel processing.

    This is a **terminal operation** that implements a fan-out pattern where
    the entire dataset is copied to each branch for independent processing.
    Each branch processes the complete dataset concurrently using separate
    transformers, and results are collected and returned in a dictionary.

    Args:
        branches: A dictionary where keys are branch names (str) and values
                  are `Transformer` instances of any subtype.
        batch_size: The number of items to batch together when sending data
                    to branches. Larger batches can improve throughput but
                    use more memory. Defaults to 1000.
        max_batch_buffer: The maximum number of batches to buffer for each
                          branch queue. Controls memory usage and creates
                          backpressure. Defaults to 1.
        use_queue_chunks: Whether to use passthrough chunking for the
                          transformers. When True, batches are processed
                          as chunks. Defaults to True.

    Returns:
        A dictionary where keys are the branch names and values are lists
        of all items processed by that branch's transformer.

    Note:
        This operation consumes the pipeline's iterator, making subsequent
        operations on the same pipeline return empty results.
    """
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

      result_iterator = transformer(stream_from_queue(), self.context_manager)  # type: ignore
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

    This makes the Pipeline compatible with Python's iterator protocol,
    allowing it to be used in for loops, list comprehensions, and other
    contexts that expect an iterable.

    Returns:
        An iterator over the processed data.

    Note:
        This operation consumes the pipeline's iterator, making subsequent
        operations on the same pipeline return empty results.
    """
    yield from self.processed_data

  def to_list(self) -> list[T]:
    """Execute the pipeline and return the results as a list.

    This is a terminal operation that consumes the pipeline's iterator
    and materializes all results into memory.

    Returns:
        A list containing all processed items from the pipeline.

    Note:
        This operation consumes the pipeline's iterator, making subsequent
        operations on the same pipeline return empty results.
    """
    return list(self.processed_data)

  def each(self, function: PipelineFunction[T]) -> None:
    """Apply a function to each element (terminal operation).

    This is a terminal operation that processes each element for side effects
    and consumes the pipeline's iterator without returning results.

    Args:
        function: The function to apply to each element. Should be used for
                  side effects like logging, updating external state, etc.

    Note:
        This operation consumes the pipeline's iterator, making subsequent
        operations on the same pipeline return empty results.
    """
    for item in self.processed_data:
      function(item)

  def first(self, n: int = 1) -> list[T]:
    """Get the first n elements of the pipeline (terminal operation).

    This is a terminal operation that consumes up to n elements from the
    pipeline's iterator and returns them as a list.

    Args:
        n: The number of elements to retrieve. Must be at least 1.

    Returns:
        A list containing the first n elements, or fewer if the pipeline
        contains fewer than n elements.

    Raises:
        AssertionError: If n is less than 1.

    Note:
        This operation partially consumes the pipeline's iterator. Subsequent
        operations will continue from where this operation left off.
    """
    assert n >= 1, "n must be at least 1"
    return list(itertools.islice(self.processed_data, n))

  def consume(self) -> None:
    """Consume the pipeline without returning results (terminal operation).

    This is a terminal operation that processes all elements in the pipeline
    for their side effects without materializing any results. Useful when
    the pipeline operations have side effects and you don't need the results.

    Note:
        This operation consumes the pipeline's iterator, making subsequent
        operations on the same pipeline return empty results.
    """
    for _ in self.processed_data:
      pass
