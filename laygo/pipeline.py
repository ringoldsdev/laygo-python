# pipeline.py
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import itertools
from multiprocessing import Manager
from queue import Queue
from typing import Any
from typing import Literal
from typing import TypeVar
from typing import overload

from loky import get_reusable_executor

from laygo.context import IContextManager
from laygo.context.parallel import ParallelContextManager
from laygo.context.types import IContextHandle
from laygo.helpers import is_context_aware
from laygo.transformers.transformer import Transformer

T = TypeVar("T")
U = TypeVar("U")
PipelineFunction = Callable[[T], Any]


# This function must be defined at the top level of the module (e.g., after imports)
def _branch_consumer_process[T](transformer: Transformer, queue: "Queue", context_handle: IContextHandle) -> list[Any]:
  """
  The entry point for a consumer process. It reconstructs the necessary
  objects and runs a dedicated pipeline instance on the data from its queue.
  """
  # Re-create the context proxy within the new process
  context_proxy = context_handle.create_proxy()

  def stream_from_queue() -> Iterator[T]:
    """A generator that yields items from the process-safe queue."""
    while (batch := queue.get()) is not None:
      yield from batch

  try:
    # Each consumer process runs its own mini-pipeline
    branch_pipeline = Pipeline(stream_from_queue(), context_manager=context_proxy)
    result_list, _ = branch_pipeline.apply(transformer).to_list()
    return result_list
  finally:
    context_proxy.shutdown()


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
    self.context_manager = context_manager if context_manager is not None else ParallelContextManager()

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
    self._user_context = ctx
    self.context_manager.update(ctx)
    return self

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
        self.processed_data = transformer(self.processed_data, context=self.context_manager)  # type: ignore
      case _ if callable(transformer):
        if is_context_aware(transformer):
          self.processed_data = transformer(self.processed_data, self.context_manager)  # type: ignore
        else:
          self.processed_data = transformer(self.processed_data)  # type: ignore
      case _:
        raise TypeError("Transformer must be a Transformer instance or a callable function")

    return self  # type: ignore

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

  def to_list(self) -> tuple[list[T], dict[str, Any]]:
    """Execute the pipeline and return the results as a list.

    This is a terminal operation that consumes the pipeline's iterator
    and materializes all results into memory.

    Returns:
        A list containing all processed items from the pipeline.

    Note:
        This operation consumes the pipeline's iterator, making subsequent
        operations on the same pipeline return empty results.
    """
    return list(self.processed_data), self.context_manager.to_dict()

  def each(self, function: PipelineFunction[T]) -> tuple[None, dict[str, Any]]:
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

    return None, self.context_manager.to_dict()

  def first(self, n: int = 1) -> tuple[list[T], dict[str, Any]]:
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
    return list(itertools.islice(self.processed_data, n)), self.context_manager.to_dict()

  def consume(self) -> tuple[None, dict[str, Any]]:
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

    return None, self.context_manager.to_dict()

  def _producer_fanout(
    self,
    source_iterator: Iterator[T],
    queues: dict[str, Queue],
    batch_size: int,
  ) -> None:
    """Producer for fan-out: sends every item to every branch."""
    for batch_tuple in itertools.batched(source_iterator, batch_size):
      batch_list = list(batch_tuple)
      for q in queues.values():
        q.put(batch_list)
    for q in queues.values():
      q.put(None)

  def _producer_router(
    self,
    source_iterator: Iterator[T],
    queues: dict[str, Queue],
    parsed_branches: list[tuple[str, Transformer, Callable]],
    batch_size: int,
  ) -> None:
    """Producer for router (`first_match=True`): sends item to the first matching branch."""
    buffers = {name: [] for name, _, _ in parsed_branches}
    for item in source_iterator:
      for name, _, condition in parsed_branches:
        if condition(item):
          branch_buffer = buffers[name]
          branch_buffer.append(item)
          if len(branch_buffer) >= batch_size:
            queues[name].put(branch_buffer)
            buffers[name] = []
          break
    for name, buffer_list in buffers.items():
      if buffer_list:
        queues[name].put(buffer_list)
    for q in queues.values():
      q.put(None)

  def _producer_broadcast(
    self,
    source_iterator: Iterator[T],
    queues: dict[str, Queue],
    parsed_branches: list[tuple[str, Transformer, Callable]],
    batch_size: int,
  ) -> None:
    """Producer for broadcast (`first_match=False`): sends item to all matching branches."""
    buffers = {name: [] for name, _, _ in parsed_branches}
    for item in source_iterator:
      item_matches = [name for name, _, condition in parsed_branches if condition(item)]

      for name in item_matches:
        buffers[name].append(item)
        branch_buffer = buffers[name]
        if len(branch_buffer) >= batch_size:
          queues[name].put(branch_buffer)
          buffers[name] = []

    for name, buffer_list in buffers.items():
      if buffer_list:
        queues[name].put(buffer_list)
    for q in queues.values():
      q.put(None)

  # In your Pipeline class

  # Overload 1: Unconditional fan-out
  @overload
  def branch(
    self,
    branches: Mapping[str, Transformer[T, Any]],
    *,
    executor_type: Literal["thread", "process"] = "thread",
    batch_size: int = 1000,
    max_batch_buffer: int = 1,
  ) -> tuple[dict[str, list[Any]], dict[str, Any]]: ...

  # Overload 2: Conditional routing
  @overload
  def branch(
    self,
    branches: Mapping[str, tuple[Transformer[T, Any], Callable[[T], bool]]],
    *,
    executor_type: Literal["thread", "process"] = "thread",
    first_match: bool = True,
    batch_size: int = 1000,
    max_batch_buffer: int = 1,
  ) -> tuple[dict[str, list[Any]], dict[str, Any]]: ...

  def branch(
    self,
    branches: Mapping[str, Transformer[T, Any]] | Mapping[str, tuple[Transformer[T, Any], Callable[[T], bool]]],
    *,
    executor_type: Literal["thread", "process"] = "thread",
    first_match: bool = True,
    batch_size: int = 1000,
    max_batch_buffer: int = 1,
  ) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """
    Forks the pipeline for parallel processing with optional conditional routing.

    This is a **terminal operation** that consumes the pipeline.

    **1. Unconditional Fan-Out:**
    If `branches` is a `Dict[str, Transformer]`, every item is sent to every branch.

    **2. Conditional Routing:**
    If `branches` is a `Dict[str, Tuple[Transformer, condition]]`, the `first_match`
    argument determines the routing logic:
    - `first_match=True` (default): Routes each item to the **first** branch
      whose condition is met. This acts as a router.
    - `first_match=False`: Routes each item to **all** branches whose
      conditions are met. This acts as a conditional broadcast.

    Args:
        branches: A dictionary defining the branches.
        executor_type: The parallelism model. 'thread' for I/O-bound tasks,
            'process' for CPU-bound tasks. Defaults to 'thread'.
        first_match: Determines the routing logic for conditional branches.
        batch_size: The number of items to batch for processing.
        max_batch_buffer: The max number of batches to buffer per branch.

    Returns:
        A tuple containing a dictionary of results and the final context.
    """
    if not branches:
      self.consume()
      return {}, {}

    first_value = next(iter(branches.values()))
    is_conditional = isinstance(first_value, tuple)

    parsed_branches: list[tuple[str, Transformer[T, Any], Callable[[T], bool]]]
    if is_conditional:
      parsed_branches = [(name, trans, cond) for name, (trans, cond) in branches.items()]  # type: ignore
    else:
      parsed_branches = [(name, trans, lambda _: True) for name, trans in branches.items()]  # type: ignore

    producer_fn: Callable
    if not is_conditional:
      producer_fn = self._producer_fanout
    elif first_match:
      producer_fn = self._producer_router
    else:
      producer_fn = self._producer_broadcast

    # Dispatch to the correct executor based on the chosen type
    if executor_type == "thread":
      return self._execute_branching_thread(
        producer_fn=producer_fn,
        parsed_branches=parsed_branches,
        batch_size=batch_size,
        max_batch_buffer=max_batch_buffer,
      )
    elif executor_type == "process":
      return self._execute_branching_process(
        producer_fn=producer_fn,
        parsed_branches=parsed_branches,
        batch_size=batch_size,
        max_batch_buffer=max_batch_buffer,
      )
    else:
      raise ValueError(f"Unsupported executor_type: '{executor_type}'. Must be 'thread' or 'process'.")

  def _execute_branching_process(
    self,
    *,
    producer_fn: Callable,
    parsed_branches: list[tuple[str, Transformer, Callable]],
    batch_size: int,
    max_batch_buffer: int,
  ) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Branching execution using a process pool for consumers."""
    source_iterator = self.processed_data
    num_branches = len(parsed_branches)
    final_results: dict[str, list[Any]] = {name: [] for name, _, _ in parsed_branches}
    context_handle = self.context_manager.get_handle()

    # A Manager creates queues that can be shared between processes
    manager = Manager()
    queues = {name: manager.Queue(maxsize=max_batch_buffer) for name, _, _ in parsed_branches}

    # The producer must run in a thread to access the pipeline's iterator,
    # while consumers run in processes for true CPU parallelism.
    producer_executor = ThreadPoolExecutor(max_workers=1)
    consumer_executor = get_reusable_executor(max_workers=num_branches)

    try:
      # Determine arguments for the producer function
      producer_args: tuple
      if producer_fn == self._producer_fanout:
        producer_args = (source_iterator, queues, batch_size)
      else:
        producer_args = (source_iterator, queues, parsed_branches, batch_size)

      # Submit the producer to the thread pool
      producer_future = producer_executor.submit(producer_fn, *producer_args)

      # Submit consumers to the process pool
      future_to_name = {
        consumer_executor.submit(_branch_consumer_process, transformer, queues[name], context_handle): name
        for name, transformer, _ in parsed_branches
      }

      # Collect results as they complete
      for future in as_completed(future_to_name):
        name = future_to_name[future]
        try:
          final_results[name] = future.result()
        except Exception:
          final_results[name] = []

      # Check for producer errors after consumers are done
      producer_future.result()

    finally:
      producer_executor.shutdown()
      # The reusable executor from loky is managed globally

    final_context = self.context_manager.to_dict()
    return final_results, final_context

  # Rename original _execute_branching to be specific
  def _execute_branching_thread(
    self,
    *,
    producer_fn: Callable,
    parsed_branches: list[tuple[str, Transformer, Callable]],
    batch_size: int,
    max_batch_buffer: int,
  ) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Shared execution logic for thread-based branching modes."""
    # ... (The original implementation of _execute_branching goes here)
    source_iterator = self.processed_data
    num_branches = len(parsed_branches)
    final_results: dict[str, list[Any]] = {name: [] for name, _, _ in parsed_branches}
    queues = {name: Queue(maxsize=max_batch_buffer) for name, _, _ in parsed_branches}

    def consumer(transformer: Transformer, queue: Queue, context_handle: IContextHandle) -> list[Any]:
      """Consumes batches from a queue and processes them."""

      def stream_from_queue() -> Iterator[T]:
        while (batch := queue.get()) is not None:
          yield from batch

      branch_pipeline = Pipeline(stream_from_queue(), context_manager=context_handle.create_proxy())  # type: ignore
      result_list, _ = branch_pipeline.apply(transformer).to_list()
      return result_list

    with ThreadPoolExecutor(max_workers=num_branches + 1) as executor:
      producer_args: tuple
      if producer_fn == self._producer_fanout:
        producer_args = (source_iterator, queues, batch_size)
      else:
        producer_args = (source_iterator, queues, parsed_branches, batch_size)
      executor.submit(producer_fn, *producer_args)

      future_to_name = {
        executor.submit(consumer, transformer, queues[name], self.context_manager.get_handle()): name
        for name, transformer, _ in parsed_branches
      }

      for future in as_completed(future_to_name):
        name = future_to_name[future]
        try:
          final_results[name] = future.result()
        except Exception:
          final_results[name] = []

    final_context = self.context_manager.to_dict()
    return final_results, final_context
