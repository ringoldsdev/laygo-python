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
  """
  Top-level function to process a single chunk.
  'loky' will use cloudpickle to serialize the 'transformer' object.
  """
  return transformer(chunk, shared_context)  # type: ignore


def createParallelTransformer[T](
  _type_hint: type[T],
  max_workers: int = 4,
  ordered: bool = True,
  chunk_size: int | None = None,
) -> "ParallelTransformer[T, T]":
  """Create a new identity parallel transformer with an explicit type hint."""
  return ParallelTransformer[T, T](
    max_workers=max_workers,
    ordered=ordered,
    chunk_size=chunk_size,
    transformer=None,
  )


class ParallelTransformer[In, Out](Transformer[In, Out]):
  """
  A transformer that executes operations concurrently using multiple processes.
  It uses 'loky' to support dynamically created transformation logic.
  """

  def __init__(
    self,
    max_workers: int = 4,
    ordered: bool = True,
    chunk_size: int | None = None,
    transformer: InternalTransformer[In, Out] | None = None,
  ):
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
      # The Pipeline's __exit__ will handle final state.
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
    """Helper to run the execution logic with a given context."""
    executor = get_reusable_executor(max_workers=self.max_workers)

    chunks_to_process = self._chunk_generator(data)
    gen_func = self._ordered_generator if self.ordered else self._unordered_generator
    processed_chunks_iterator = gen_func(chunks_to_process, executor, shared_context)

    for result_chunk in processed_chunks_iterator:
      yield from result_chunk

  # ... The rest of the file remains the same ...
  def _ordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    executor: ProcessPoolExecutor,
    shared_context: MutableMapping[str, Any],
  ) -> Iterator[list[Out]]:
    """Generate results in their original order."""
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
    """Generate results as they complete."""
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

  def tap(self, function: PipelineFunction[Out, Any]) -> "ParallelTransformer[In, Out]":
    super().tap(function)
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
