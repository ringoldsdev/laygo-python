from collections import deque
from collections.abc import Iterator
from concurrent.futures import wait
import itertools

from loky import as_completed
from loky import get_reusable_executor

from laygo.context.types import IContextHandle
from laygo.context.types import IContextManager
from laygo.types import ExecutionStrategy
from laygo.types import InternalTransformer


def _worker_process_chunk[In, Out](
  transformer_logic: InternalTransformer[In, Out],
  context_handle: IContextHandle,
  chunk: list[In],
) -> list[Out]:
  """Top-level function executed by each worker process."""
  context_proxy = context_handle.create_proxy()
  try:
    return transformer_logic(chunk, context_proxy)
  finally:
    context_proxy.shutdown()


class ProcessStrategy[In, Out](ExecutionStrategy[In, Out]):
  """Execute transformer logic using a process pool."""

  def __init__(self, max_workers: int = 4, ordered: bool = True):
    self.max_workers = max_workers
    self.ordered = ordered

  def execute(
    self,
    transformer_logic: InternalTransformer[In, Out],
    chunks: Iterator[list[In]],
    context: IContextManager,
  ) -> Iterator[list[Out]]:
    """Execute the transformer by distributing chunks to a process pool."""
    context_handle = context.get_handle()
    executor = get_reusable_executor(max_workers=self.max_workers)

    gen_func = self._ordered_generator if self.ordered else self._unordered_generator
    yield from gen_func(chunks, transformer_logic, executor, context_handle)

  def _ordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    transformer: InternalTransformer[In, Out],
    executor,
    context_handle: IContextHandle,
  ) -> Iterator[list[Out]]:
    """Generate results in their original order, with robust error handling."""
    futures = deque()
    chunks_iter = iter(chunks_iter)

    # Submit the initial batch of tasks
    for _ in range(self.max_workers + 1):
      try:
        chunk = next(chunks_iter)
        futures.append(executor.submit(_worker_process_chunk, transformer, context_handle, chunk))
      except StopIteration:
        break

    try:
      while futures:
        result = futures.popleft().result()

        try:
          chunk = next(chunks_iter)
          futures.append(executor.submit(_worker_process_chunk, transformer, context_handle, chunk))
        except StopIteration:
          pass

        yield result
    finally:
      for future in futures:
        future.cancel()
      if futures:
        wait(list(futures))

  def _unordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    transformer: InternalTransformer[In, Out],
    executor,
    context_handle: IContextHandle,
  ) -> Iterator[list[Out]]:
    """Generate results as they complete, with robust error handling."""
    futures = {
      executor.submit(_worker_process_chunk, transformer, context_handle, chunk)
      for chunk in itertools.islice(chunks_iter, self.max_workers + 1)
    }

    try:
      for future in as_completed(futures):
        result = future.result()
        futures.remove(future)

        try:
          chunk = next(chunks_iter)
          futures.add(executor.submit(_worker_process_chunk, transformer, context_handle, chunk))
        except StopIteration:
          pass

        yield result
    finally:
      for future in futures:
        future.cancel()
      if futures:
        wait(futures)
