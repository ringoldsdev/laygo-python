from collections import deque
from collections.abc import Iterator
from concurrent.futures import wait
import itertools

from loky import as_completed
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
        # Get the result of the oldest task. If it failed or the pool
        # is broken, .result() will raise an exception.
        result = futures.popleft().result()

        # If successful, submit a new task.
        try:
          chunk = next(chunks_iter)
          futures.append(executor.submit(_worker_process_chunk, transformer, context_handle, chunk))
        except StopIteration:
          # No more chunks to process.
          pass

        yield result
    finally:
      # This cleanup runs if the loop finishes or if an exception occurs.
      # It prevents orphaned processes by cancelling pending tasks.
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
      # as_completed is ideal for this "process as they finish" pattern
      for future in as_completed(futures):
        # Get the result. This raises an exception if the task failed,
        # which immediately stops the loop and proceeds to finally.
        result = future.result()

        # Remove the completed future from our tracking set
        futures.remove(future)

        # Try to submit a new task to replace the one that just finished
        try:
          chunk = next(chunks_iter)
          futures.add(executor.submit(_worker_process_chunk, transformer, context_handle, chunk))
        except StopIteration:
          # No more chunks left to submit.
          pass

        yield result
    finally:
      # Clean up any futures that were still running or pending when
      # an exception occurred or the input was exhausted.
      for future in futures:
        future.cancel()
      if futures:
        wait(futures)
