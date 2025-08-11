from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import wait
import itertools

from laygo.context.types import IContextManager
from laygo.transformers.strategies.types import ExecutionStrategy
from laygo.transformers.types import InternalTransformer


class ThreadedStrategy[In, Out](ExecutionStrategy[In, Out]):
  """Execute transformer logic using a thread pool."""

  def __init__(self, max_workers: int = 4, ordered: bool = True):
    """Initialize the threaded strategy.

    Args:
        max_workers: Maximum number of worker threads.
        ordered: Whether to preserve order of results.
    """
    self.max_workers = max_workers
    self.ordered = ordered

  def execute(
    self,
    transformer_logic: InternalTransformer[In, Out],
    chunks: Iterator[list[In]],
    context: IContextManager,
  ) -> Iterator[list[Out]]:
    """Execute the transformer by distributing chunks to a thread pool.

    Args:
        transformer_logic: The transformation function to apply.
        chunks: Iterator of pre-chunked data.
        context: Context manager for the execution.

    Returns:
        Iterator of transformed chunks.
    """
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
      gen_func = self._ordered_generator if self.ordered else self._unordered_generator
      yield from gen_func(chunks, transformer_logic, executor, context)

  def _ordered_generator(
    self,
    chunks_iter: Iterator[list[In]],
    transformer: InternalTransformer[In, Out],
    executor: ThreadPoolExecutor,
    context: IContextManager,
  ) -> Iterator[list[Out]]:
    """Generate results in their original order, with robust error handling.

    Args:
        chunks_iter: Iterator of chunks to process.
        transformer: The transformation function to apply.
        executor: Thread pool executor.
        context: Context manager for the execution.

    Returns:
        Iterator of transformed chunks in original order.
    """
    futures = deque()
    chunks_iter = iter(chunks_iter)

    # Submit the initial batch of tasks
    for _ in range(self.max_workers + 1):
      try:
        chunk = next(chunks_iter)
        futures.append(executor.submit(transformer, chunk, context))
      except StopIteration:
        break

    try:
      while futures:
        result = futures.popleft().result()

        try:
          chunk = next(chunks_iter)
          futures.append(executor.submit(transformer, chunk, context))
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
    executor: ThreadPoolExecutor,
    context: IContextManager,
  ) -> Iterator[list[Out]]:
    """Generate results as they complete, with robust error handling.

    Args:
        chunks_iter: Iterator of chunks to process.
        transformer: The transformation function to apply.
        executor: Thread pool executor.
        context: Context manager for the execution.

    Returns:
        Iterator of transformed chunks as they complete.
    """
    futures = {
      executor.submit(transformer, chunk, context) for chunk in itertools.islice(chunks_iter, self.max_workers + 1)
    }

    try:
      for future in as_completed(futures):
        result = future.result()
        futures.remove(future)

        try:
          chunk = next(chunks_iter)
          futures.add(executor.submit(transformer, chunk, context))
        except StopIteration:
          pass

        yield result
    finally:
      for future in futures:
        future.cancel()
      if futures:
        wait(futures)
