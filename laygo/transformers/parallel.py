"""Parallel transformer implementation using multiple processes and loky."""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import MutableMapping
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import wait
import copy
import itertools
import multiprocessing as mp
from multiprocessing.managers import DictProxy
from typing import Any
from typing import Union
from typing import overload

from loky import ProcessPoolExecutor
from loky import get_reusable_executor

from laygo.errors import ErrorHandler
from laygo.helpers import PipelineContext
from laygo.transformers.transformer import ChunkErrorHandler
from laygo.transformers.transformer import InternalTransformer
from laygo.transformers.transformer import PipelineFunction
from laygo.transformers.transformer import Transformer


def _process_chunk_for_multiprocessing[In, Out](
  transformer: InternalTransformer[In, Out],
  shared_context: MutableMapping[str, Any],
  chunk: list[In],
) -> list[Out]:
  """Process a single chunk at the top level.

  This function is designed to work with 'loky' which uses cloudpickle
  to serialize the 'transformer' object.

  Args:
      transformer: The transformation function to apply.
      shared_context: The shared context for processing.
      chunk: The data chunk to process.

  Returns:
      The processed chunk.
  """
  return transformer(chunk, shared_context)  # type: ignore


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

  def __call__(self, data: Iterable[In], context: PipelineContext | None = None) -> Iterator[Out]:
    """Execute the transformer on data concurrently.

    It uses the shared context provided by the Pipeline, if available.

    Args:
        data: The input data to process.
        context: Optional pipeline context for shared state.

    Returns:
        An iterator over the transformed data.
    """
    run_context = context if context is not None else self.context

    # Detect if the context is already managed by the Pipeline.
    is_managed_context = isinstance(run_context, DictProxy)

    if is_managed_context:
      # Use the existing shared context and lock from the Pipeline.
      shared_context = run_context
      yield from self._execute_with_context(data, shared_context)
    else:
      # Fallback for standalone use: create a temporary manager.
      with mp.Manager() as manager:
        initial_ctx_data = dict(run_context)
        shared_context = manager.dict(initial_ctx_data)
        if "lock" not in shared_context:
          shared_context["lock"] = manager.Lock()

        yield from self._execute_with_context(data, shared_context)

        # Copy results back to the original non-shared context.
        final_context_state = dict(shared_context)
        final_context_state.pop("lock", None)
        run_context.update(final_context_state)

  def _execute_with_context(self, data: Iterable[In], shared_context: MutableMapping[str, Any]) -> Iterator[Out]:
    """Execute the transformation logic with a given context.

    Args:
        data: The input data to process.
        shared_context: The shared context for the execution.

    Returns:
        An iterator over the transformed data.
    """
    executor = get_reusable_executor(max_workers=self.max_workers)

    chunks_to_process = self._chunk_generator(data)
    gen_func = self._ordered_generator if self.ordered else self._unordered_generator
    processed_chunks_iterator = gen_func(chunks_to_process, executor, shared_context)

    for result_chunk in processed_chunks_iterator:
      yield from result_chunk

  def _ordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    executor: ProcessPoolExecutor,
    shared_context: MutableMapping[str, Any],
  ) -> Iterator[list[Out]]:
    """Generate results in their original order.

    Args:
        chunks_iter: Iterator over data chunks.
        executor: The process pool executor.
        shared_context: The shared context for processing.

    Returns:
        An iterator over processed chunks in order.
    """
    futures: deque[Future[list[Out]]] = deque()
    for _ in range(self.max_workers + 1):
      try:
        chunk = next(chunks_iter)
        futures.append(
          executor.submit(
            _process_chunk_for_multiprocessing,
            self.transformer,
            shared_context,
            chunk,
          )
        )
      except StopIteration:
        break
    while futures:
      yield futures.popleft().result()
      try:
        chunk = next(chunks_iter)
        futures.append(
          executor.submit(
            _process_chunk_for_multiprocessing,
            self.transformer,
            shared_context,
            chunk,
          )
        )
      except StopIteration:
        continue

  def _unordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    executor: ProcessPoolExecutor,
    shared_context: MutableMapping[str, Any],
  ) -> Iterator[list[Out]]:
    """Generate results as they complete.

    Args:
        chunks_iter: Iterator over data chunks.
        executor: The process pool executor.
        shared_context: The shared context for processing.

    Returns:
        An iterator over processed chunks as they complete.
    """
    futures = {
      executor.submit(
        _process_chunk_for_multiprocessing,
        self.transformer,
        shared_context,
        chunk,
      )
      for chunk in itertools.islice(chunks_iter, self.max_workers + 1)
    }
    while futures:
      done, futures = wait(futures, return_when=FIRST_COMPLETED)
      for future in done:
        yield future.result()
        try:
          chunk = next(chunks_iter)
          futures.add(
            executor.submit(
              _process_chunk_for_multiprocessing,
              self.transformer,
              shared_context,
              chunk,
            )
          )
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

  def short_circuit(self, function: Callable[[PipelineContext], bool | None]) -> "ParallelTransformer[In, Out]":
    super().short_circuit(function)
    return self
