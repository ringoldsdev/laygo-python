"""Parallel transformer implementation using multiple threads."""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import copy
from functools import partial
import itertools
from typing import Any
from typing import Union
from typing import overload

from laygo.context import IContextManager
from laygo.context import ParallelContextManager
from laygo.errors import ErrorHandler
from laygo.transformers.transformer import DEFAULT_CHUNK_SIZE
from laygo.transformers.transformer import ChunkErrorHandler
from laygo.transformers.transformer import InternalTransformer
from laygo.transformers.transformer import PipelineFunction
from laygo.transformers.transformer import Transformer


def createThreadedTransformer[T](
  _type_hint: type[T],
  max_workers: int = 4,
  ordered: bool = True,
  chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> "ThreadedTransformer[T, T]":
  """Create a new identity threaded transformer with an explicit type hint.

  Args:
      _type_hint: Type hint for the data being processed.
      max_workers: Maximum number of worker threads.
      ordered: Whether to preserve order of results.
      chunk_size: Size of chunks to process data in.

  Returns:
      A new identity threaded transformer.
  """
  return ThreadedTransformer[T, T](
    max_workers=max_workers,
    ordered=ordered,
    chunk_size=chunk_size,
    transformer=None,
  )


class ThreadedTransformer[In, Out](Transformer[In, Out]):
  """A transformer that executes operations concurrently using multiple threads.

  This transformer processes data chunks in parallel using a thread pool,
  which is effective for I/O-bound operations but may be limited by the
  Global Interpreter Lock (GIL) for CPU-bound tasks.
  """

  def __init__(
    self,
    max_workers: int = 4,
    ordered: bool = True,
    chunk_size: int | None = None,
    transformer: InternalTransformer[In, Out] | None = None,
  ) -> None:
    """Initialize the threaded transformer.

    Args:
        max_workers: Maximum number of worker threads.
        ordered: If True, results are yielded in order. If False, results
                 are yielded as they complete.
        chunk_size: Size of data chunks to process.
        transformer: The transformation logic chain.
    """
    super().__init__(chunk_size, transformer)
    self.max_workers = max_workers
    self.ordered = ordered
    # Rule 3: Threaded transformers create a parallel context manager by default.
    # This is because threads share memory, so a thread-safe (locking) manager is required.
    self._default_context = ParallelContextManager()

  @classmethod
  def from_transformer[T, U](
    cls,
    transformer: Transformer[T, U],
    chunk_size: int | None = None,
    max_workers: int = 4,
    ordered: bool = True,
  ) -> "ThreadedTransformer[T, U]":
    """Create a ThreadedTransformer from an existing Transformer's logic.

    Args:
        transformer: The base transformer to copy the transformation logic from.
        chunk_size: Optional chunk size override.
        max_workers: Maximum number of worker threads.
        ordered: If True, results are yielded in order.

    Returns:
        A new ThreadedTransformer with the same transformation logic.
    """
    return cls(
      chunk_size=chunk_size or transformer.chunk_size,
      transformer=copy.deepcopy(transformer.transformer),  # type: ignore
      max_workers=max_workers,
      ordered=ordered,
    )

  def __call__(self, data: Iterable[In], context: IContextManager | None = None) -> Iterator[Out]:
    """Execute the transformer on data concurrently.

    It uses the shared context provided by the Pipeline, if available.

    Args:
        data: The input data to process.
        context: Optional pipeline context for shared state.

    Returns:
        An iterator over the transformed data.
    """
    run_context = context or self._default_context

    # Since threads share memory, we can pass the context manager directly.
    # No handle/proxy mechanism is needed, but the locking inside
    # ParallelContextManager is crucial for thread safety.
    try:
      yield from self._execute_with_context(data, run_context)
    finally:
      if run_context is self._default_context:
        self._default_context.shutdown()

  def _execute_with_context(self, data: Iterable[In], shared_context: IContextManager) -> Iterator[Out]:
    """Execute the transformation logic with a given context.

    Args:
        data: The input data to process.
        shared_context: The shared context for the execution.

    Returns:
        An iterator over the transformed data.
    """

    def process_chunk(chunk: list[In], shared_context: IContextManager) -> list[Out]:
      """Process a single chunk by passing the chunk and context explicitly.

      Args:
          chunk: The data chunk to process.
          shared_context: The shared context for processing.

      Returns:
          The processed chunk.
      """
      return self.transformer(chunk, shared_context)  # type: ignore

    # Create a partial function with the shared_context "baked in".
    process_chunk_with_context = partial(process_chunk, shared_context=shared_context)

    def _ordered_generator(chunks_iter: Iterator[list[In]], executor: ThreadPoolExecutor) -> Iterator[list[Out]]:
      """Generate results in their original order."""
      futures: deque[Future[list[Out]]] = deque()
      for _ in range(self.max_workers + 1):
        try:
          chunk = next(chunks_iter)
          futures.append(executor.submit(process_chunk_with_context, chunk))
        except StopIteration:
          break
      while futures:
        yield futures.popleft().result()
        try:
          chunk = next(chunks_iter)
          futures.append(executor.submit(process_chunk_with_context, chunk))
        except StopIteration:
          continue

    def _unordered_generator(chunks_iter: Iterator[list[In]], executor: ThreadPoolExecutor) -> Iterator[list[Out]]:
      """Generate results as they complete."""
      futures = {
        executor.submit(process_chunk_with_context, chunk)
        for chunk in itertools.islice(chunks_iter, self.max_workers + 1)
      }
      while futures:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
          yield future.result()
          try:
            chunk = next(chunks_iter)
            futures.add(executor.submit(process_chunk_with_context, chunk))
          except StopIteration:
            continue

    def result_iterator_manager() -> Iterator[Out]:
      """Manage the thread pool and yield flattened results."""
      with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        chunks_to_process = self._chunk_generator(data)
        gen_func = _ordered_generator if self.ordered else _unordered_generator
        processed_chunks_iterator = gen_func(chunks_to_process, executor)
        for result_chunk in processed_chunks_iterator:
          yield from result_chunk

    return result_iterator_manager()

  # --- Overridden Chaining Methods to Preserve Type ---

  def on_error(self, handler: ChunkErrorHandler[In, Out] | ErrorHandler) -> "ThreadedTransformer[In, Out]":
    super().on_error(handler)
    return self

  def map[U](self, function: PipelineFunction[Out, U]) -> "ThreadedTransformer[In, U]":
    super().map(function)
    return self  # type: ignore

  def filter(self, predicate: PipelineFunction[Out, bool]) -> "ThreadedTransformer[In, Out]":
    super().filter(predicate)
    return self

  @overload
  def flatten[T](self: "ThreadedTransformer[In, list[T]]") -> "ThreadedTransformer[In, T]": ...
  @overload
  def flatten[T](self: "ThreadedTransformer[In, tuple[T, ...]]") -> "ThreadedTransformer[In, T]": ...
  @overload
  def flatten[T](self: "ThreadedTransformer[In, set[T]]") -> "ThreadedTransformer[In, T]": ...
  def flatten[T](  # type: ignore
    self: Union[
      "ThreadedTransformer[In, list[T]]", "ThreadedTransformer[In, tuple[T, ...]]", "ThreadedTransformer[In, set[T]]"
    ],
  ) -> "ThreadedTransformer[In, T]":
    super().flatten()  # type: ignore
    return self  # type: ignore

  def tap(self, arg: Union["Transformer[Out, Any]", PipelineFunction[Out, Any]]) -> "ThreadedTransformer[In, Out]":
    super().tap(arg)
    return self

  def apply[T](
    self, t: Callable[["ThreadedTransformer[In, Out]"], "Transformer[In, T]"]
  ) -> "ThreadedTransformer[In, T]":
    super().apply(t)  # type: ignore
    return self  # type: ignore

  def catch[U](
    self,
    sub_pipeline_builder: Callable[[Transformer[Out, Out]], Transformer[Out, U]],
    on_error: ChunkErrorHandler[Out, U] | None = None,
  ) -> "ThreadedTransformer[In, U]":
    super().catch(sub_pipeline_builder, on_error)
    return self  # type: ignore

  def short_circuit(self, function: Callable[[IContextManager], bool | None]) -> "ThreadedTransformer[In, Out]":
    super().short_circuit(function)
    return self
