"""Parallel transformer implementation using multiple threads."""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import MutableMapping
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import copy
from functools import partial
import itertools
from multiprocessing.managers import DictProxy
import threading
from typing import Any
from typing import Union
from typing import overload

from laygo.errors import ErrorHandler
from laygo.helpers import PipelineContext
from laygo.transformers.transformer import DEFAULT_CHUNK_SIZE
from laygo.transformers.transformer import ChunkErrorHandler
from laygo.transformers.transformer import InternalTransformer
from laygo.transformers.transformer import PipelineFunction
from laygo.transformers.transformer import Transformer


class ThreadedPipelineContextType(PipelineContext):
  """A specific context type for threaded transformers that includes a lock."""

  lock: threading.Lock


def createThreadedTransformer[T](
  _type_hint: type[T],
  max_workers: int = 4,
  ordered: bool = True,
  chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> "ThreadedTransformer[T, T]":
  """Create a new identity threaded transformer with an explicit type hint."""
  return ThreadedTransformer[T, T](
    max_workers=max_workers,
    ordered=ordered,
    chunk_size=chunk_size,
    transformer=None,
  )


class ThreadedTransformer[In, Out](Transformer[In, Out]):
  """
  A transformer that executes operations concurrently using multiple threads.
  """

  def __init__(
    self,
    max_workers: int = 4,
    ordered: bool = True,
    chunk_size: int | None = None,
    transformer: InternalTransformer[In, Out] | None = None,
  ):
    """
    Initialize the threaded transformer.

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

  @classmethod
  def from_transformer[T, U](
    cls,
    transformer: Transformer[T, U],
    chunk_size: int | None = None,
    max_workers: int = 4,
    ordered: bool = True,
  ) -> "ThreadedTransformer[T, U]":
    """
    Create a ThreadedTransformer from an existing Transformer's logic.

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

  def __call__(self, data: Iterable[In], context: PipelineContext | None = None) -> Iterator[Out]:
    """
    Executes the transformer on data concurrently. It uses the shared
    context provided by the Pipeline, if available.
    """
    run_context = context if context is not None else self.context

    # Detect if the context is already managed by the Pipeline.
    is_managed_context = isinstance(run_context, DictProxy)

    if is_managed_context:
      # Use the existing shared context and lock from the Pipeline.
      shared_context = run_context
      yield from self._execute_with_context(data, shared_context)
      # The context is live, so no need to update it here.
      # The Pipeline's __del__ will handle final state.
    else:
      # Fallback for standalone use: create a thread-safe context.
      # Since threads share memory, we can use the context directly with a lock.
      if "lock" not in run_context:
        run_context["lock"] = threading.Lock()

      yield from self._execute_with_context(data, run_context)
      # Context is already updated in-place for threads (shared memory)

  def _execute_with_context(self, data: Iterable[In], shared_context: MutableMapping[str, Any]) -> Iterator[Out]:
    """Helper to run the execution logic with a given context."""

    def process_chunk(chunk: list[In], shared_context: MutableMapping[str, Any]) -> list[Out]:
      """
      Process a single chunk by passing the chunk and context explicitly
      to the transformer chain. This is safer and avoids mutating self.
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

  def tap(self, function: PipelineFunction[Out, Any]) -> "ThreadedTransformer[In, Out]":
    super().tap(function)
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

  def short_circuit(self, function: Callable[[PipelineContext], bool | None]) -> "ThreadedTransformer[In, Out]":
    super().short_circuit(function)
    return self
