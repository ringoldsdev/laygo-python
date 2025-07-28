"""Parallel transformer implementation using multiple processes and loky."""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import wait
import copy
import itertools
from typing import Any
from typing import Union
from typing import overload

from loky import get_reusable_executor

from laygo.context import ParallelContextManager
from laygo.context.types import IContextHandle
from laygo.context.types import IContextManager
from laygo.errors import ErrorHandler
from laygo.transformers.transformer import ChunkErrorHandler
from laygo.transformers.transformer import InternalTransformer
from laygo.transformers.transformer import PipelineFunction
from laygo.transformers.transformer import Transformer


def _worker_process_chunk[In, Out](
  transformer_logic: InternalTransformer[In, Out],
  context_handle: IContextHandle,
  chunk: list[In],
) -> list[Out]:
  """
  Top-level function executed by each worker process.
  It reconstructs the context proxy from the handle and runs the transformation.
  """
  context_proxy = context_handle.create_proxy()
  try:
    return transformer_logic(chunk, context_proxy)
  finally:
    # The proxy's shutdown is a no-op, but it's good practice to call it.
    context_proxy.shutdown()


def createParallelTransformer[T](
  _type_hint: type[T],
  max_workers: int = 4,
  ordered: bool = True,
  chunk_size: int | None = None,
) -> "ParallelTransformer[T, T]":
  """Create a new identity parallel transformer with an explicit type hint.

  Args:
      _type_hint: Type hint for the data being processed.
      max_workers: Maximum number of worker processes.
      ordered: Whether to preserve order of results.
      chunk_size: Size of chunks to process data in.

  Returns:
      A new identity parallel transformer.
  """
  return ParallelTransformer[T, T](
    max_workers=max_workers,
    ordered=ordered,
    chunk_size=chunk_size,
    transformer=None,
  )


class ParallelTransformer[In, Out](Transformer[In, Out]):
  """A transformer that executes operations concurrently using multiple processes.

  This transformer uses 'loky' to support dynamically created transformation
  logic and provides true parallelism by bypassing Python's Global Interpreter
  Lock (GIL). It's ideal for CPU-bound operations.
  """

  def __init__(
    self,
    max_workers: int = 4,
    ordered: bool = True,
    chunk_size: int | None = None,
    transformer: InternalTransformer[In, Out] | None = None,
  ) -> None:
    """Initialize the parallel transformer.

    Args:
        max_workers: Maximum number of worker processes.
        ordered: If True, results are yielded in order. If False, results
                 are yielded as they complete.
        chunk_size: Size of data chunks to process.
        transformer: The transformation logic chain.
    """
    super().__init__(chunk_size, transformer)
    self.max_workers = max_workers
    self.ordered = ordered
    # Rule 3: Parallel transformers create a parallel context manager by default.
    self._default_context = ParallelContextManager()

  @classmethod
  def from_transformer[T, U](
    cls,
    transformer: Transformer[T, U],
    chunk_size: int | None = None,
    max_workers: int = 4,
    ordered: bool = True,
  ) -> "ParallelTransformer[T, U]":
    """Create a ParallelTransformer from an existing Transformer's logic.

    Args:
        transformer: The base transformer to copy the transformation logic from.
        chunk_size: Optional chunk size override.
        max_workers: Maximum number of worker processes.
        ordered: If True, results are yielded in order.

    Returns:
        A new ParallelTransformer with the same transformation logic.
    """
    return cls(
      chunk_size=chunk_size or transformer.chunk_size,
      transformer=copy.deepcopy(transformer.transformer),  # type: ignore
      max_workers=max_workers,
      ordered=ordered,
    )

  def __call__(self, data: Iterable[In], context: IContextManager | None = None) -> Iterator[Out]:
    """Execute the transformer by distributing chunks to a process pool."""
    run_context = context if context is not None else self._default_context

    # Get the picklable handle from the context manager.
    context_handle = run_context.get_handle()

    executor = get_reusable_executor(max_workers=self.max_workers)
    chunks_to_process = self._chunk_generator(data)

    gen_func = self._ordered_generator if self.ordered else self._unordered_generator

    try:
      processed_chunks_iterator = gen_func(chunks_to_process, executor, context_handle)
      for result_chunk in processed_chunks_iterator:
        yield from result_chunk
    finally:
      if run_context is self._default_context:
        self._default_context.shutdown()

  def _ordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    executor,
    context_handle: IContextHandle,
  ) -> Iterator[list[Out]]:
    """Generate results in their original order."""
    futures = deque()
    for _ in range(self.max_workers + 1):
      try:
        chunk = next(chunks_iter)
        futures.append(executor.submit(_worker_process_chunk, self.transformer, context_handle, chunk))
      except StopIteration:
        break
    while futures:
      yield futures.popleft().result()
      try:
        chunk = next(chunks_iter)
        futures.append(executor.submit(_worker_process_chunk, self.transformer, context_handle, chunk))
      except StopIteration:
        continue

  def _unordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    executor,
    context_handle: IContextHandle,
  ) -> Iterator[list[Out]]:
    """Generate results as they complete."""
    futures = {
      executor.submit(_worker_process_chunk, self.transformer, context_handle, chunk)
      for chunk in itertools.islice(chunks_iter, self.max_workers + 1)
    }
    while futures:
      done, futures = wait(futures, return_when=FIRST_COMPLETED)
      for future in done:
        yield future.result()
        try:
          chunk = next(chunks_iter)
          futures.add(executor.submit(_worker_process_chunk, self.transformer, context_handle, chunk))
        except StopIteration:
          continue

  def on_error(self, handler: ChunkErrorHandler[In, Out] | ErrorHandler) -> "ParallelTransformer[In, Out]":
    super().on_error(handler)
    return self

  def map[U](self, function: PipelineFunction[Out, U]) -> "ParallelTransformer[In, U]":
    super().map(function)
    return self  # type: ignore

  def filter(self, predicate: PipelineFunction[Out, bool]) -> "ParallelTransformer[In, Out]":
    super().filter(predicate)
    return self

  @overload
  def flatten[T](self: "ParallelTransformer[In, list[T]]") -> "ParallelTransformer[In, T]": ...
  @overload
  def flatten[T](self: "ParallelTransformer[In, tuple[T, ...]]") -> "ParallelTransformer[In, T]": ...
  @overload
  def flatten[T](self: "ParallelTransformer[In, set[T]]") -> "ParallelTransformer[In, T]": ...
  def flatten[T](  # type: ignore
    self: Union[
      "ParallelTransformer[In, list[T]]",
      "ParallelTransformer[In, tuple[T, ...]]",
      "ParallelTransformer[In, set[T]]",
    ],
  ) -> "ParallelTransformer[In, T]":
    super().flatten()  # type: ignore
    return self  # type: ignore

  def tap(self, arg: Union["Transformer[Out, Any]", PipelineFunction[Out, Any]]) -> "ParallelTransformer[In, Out]":
    super().tap(arg)
    return self

  def apply[T](
    self, t: Callable[["ParallelTransformer[In, Out]"], "Transformer[In, T]"]
  ) -> "ParallelTransformer[In, T]":
    super().apply(t)  # type: ignore
    return self  # type: ignore

  def catch[U](
    self,
    sub_pipeline_builder: Callable[[Transformer[Out, Out]], Transformer[Out, U]],
    on_error: ChunkErrorHandler[Out, U] | None = None,
  ) -> "ParallelTransformer[In, U]":
    super().catch(sub_pipeline_builder, on_error)
    return self  # type: ignore

  def short_circuit(self, function: Callable[[IContextManager], bool | None]) -> "ParallelTransformer[In, Out]":
    super().short_circuit(function)
    return self
