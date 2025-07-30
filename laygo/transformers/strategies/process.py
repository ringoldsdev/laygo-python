from collections import deque
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import wait
import itertools

from loky import get_reusable_executor

from laygo.context.types import IContextHandle
from laygo.transformers.strategies.types import ExecutionStrategy
from laygo.transformers.types import InternalTransformer


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


class ProcessStrategy[In, Out](ExecutionStrategy[In, Out]):
  def __init__(self, max_workers: int = 4, ordered: bool = True):
    self.max_workers = max_workers
    self.ordered = ordered

  def execute(self, transformer_logic, chunk_generator, data, context):
    """Execute the transformer by distributing chunks to a process pool."""

    # Get the picklable handle from the context manager.
    context_handle = context.get_handle()

    executor = get_reusable_executor(max_workers=self.max_workers)
    chunks_to_process = chunk_generator(data)

    gen_func = self._ordered_generator if self.ordered else self._unordered_generator

    processed_chunks_iterator = gen_func(chunks_to_process, transformer_logic, executor, context_handle)
    for result_chunk in processed_chunks_iterator:
      yield from result_chunk

  def _ordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    transformer: InternalTransformer[In, Out],
    executor,
    context_handle: IContextHandle,
  ) -> Iterator[list[Out]]:
    """Generate results in their original order."""
    futures = deque()
    for _ in range(self.max_workers + 1):
      try:
        chunk = next(chunks_iter)
        futures.append(executor.submit(_worker_process_chunk, transformer, context_handle, chunk))
      except StopIteration:
        break
    while futures:
      yield futures.popleft().result()
      try:
        chunk = next(chunks_iter)
        futures.append(executor.submit(_worker_process_chunk, transformer, context_handle, chunk))
      except StopIteration:
        continue

  def _unordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    transformer: InternalTransformer[In, Out],
    executor,
    context_handle: IContextHandle,
  ) -> Iterator[list[Out]]:
    """Generate results as they complete."""
    futures = {
      executor.submit(_worker_process_chunk, transformer, context_handle, chunk)
      for chunk in itertools.islice(chunks_iter, self.max_workers + 1)
    }
    while futures:
      done, futures = wait(futures, return_when=FIRST_COMPLETED)
      for future in done:
        yield future.result()
        try:
          chunk = next(chunks_iter)
          futures.add(executor.submit(_worker_process_chunk, transformer, context_handle, chunk))
        except StopIteration:
          continue
