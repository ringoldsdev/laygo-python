from collections import deque
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import itertools
import threading
from typing import ClassVar

from laygo.context.types import IContextManager
from laygo.transformers.strategies.types import ChunkGenerator
from laygo.transformers.strategies.types import ExecutionStrategy
from laygo.transformers.types import InternalTransformer


class ThreadedStrategy[In, Out](ExecutionStrategy[In, Out]):
  # Class-level thread pool cache to reuse executors
  _thread_pools: ClassVar[dict[int, ThreadPoolExecutor]] = {}
  _pool_lock: ClassVar[threading.Lock] = threading.Lock()

  def __init__(self, max_workers: int = 4, ordered: bool = True):
    self.max_workers = max_workers
    self.ordered = ordered

  @classmethod
  def _get_thread_pool(cls, max_workers: int) -> ThreadPoolExecutor:
    """Get or create a reusable thread pool for the given worker count."""
    with cls._pool_lock:
      if max_workers not in cls._thread_pools:
        cls._thread_pools[max_workers] = ThreadPoolExecutor(
          max_workers=max_workers, thread_name_prefix=f"laygo-{max_workers}"
        )
      return cls._thread_pools[max_workers]

  def execute(self, transformer_logic, chunk_generator, data, context):
    """Execute the transformer on data concurrently.

    Uses a reusable thread pool to minimize thread creation overhead.

    Args:
        transformer_logic: The transformation function to apply.
        chunk_generator: Function to generate data chunks.
        data: The input data to process.
        context: Optional pipeline context for shared state.

    Returns:
        An iterator over the transformed data.
    """
    yield from self._execute_with_context(data, transformer_logic, context, chunk_generator)

  def _execute_with_context(
    self,
    data: Iterable[In],
    transformer: InternalTransformer[In, Out],
    shared_context: IContextManager,
    chunk_generator: ChunkGenerator[In],
  ) -> Iterator[Out]:
    """Execute the transformation logic with a given context.

    Args:
        data: The input data to process.
        transformer: The transformation function to apply.
        shared_context: The shared context for the execution.
        chunk_generator: Function to generate data chunks.

    Returns:
        An iterator over the transformed data.
    """

    def process_chunk(chunk: list[In]) -> list[Out]:
      """Process a single chunk by passing the chunk and context explicitly.

      Args:
          chunk: The data chunk to process.

      Returns:
          The processed chunk.
      """
      return transformer(chunk, shared_context)

    def _ordered_generator(chunks_iter: Iterator[list[In]], executor: ThreadPoolExecutor) -> Iterator[list[Out]]:
      """Generate results in their original order."""
      futures: deque[Future[list[Out]]] = deque()
      executor_shutdown = False

      # Pre-submit initial batch of futures
      for _ in range(min(self.max_workers, 10)):
        if executor_shutdown:
          break
        try:
          chunk = next(chunks_iter)
          futures.append(executor.submit(process_chunk, chunk))
        except StopIteration:
          break
        except RuntimeError as e:
          if "cannot schedule new futures after shutdown" in str(e):
            executor_shutdown = True
            break
          raise

      while futures:
        try:
          # Get the next result
          result = futures.popleft().result()
          yield result

          # Try to submit the next chunk only if executor is not shutdown
          if not executor_shutdown:
            try:
              chunk = next(chunks_iter)
              futures.append(executor.submit(process_chunk, chunk))
            except StopIteration:
              continue
            except RuntimeError as e:
              if "cannot schedule new futures after shutdown" in str(e):
                executor_shutdown = True
                continue
              raise
        except Exception:
          # Cancel remaining futures and re-raise
          for future in futures:
            try:
              future.cancel()
            except Exception:
              pass  # Ignore cancellation errors
          futures.clear()
          raise

    def _unordered_generator(chunks_iter: Iterator[list[In]], executor: ThreadPoolExecutor) -> Iterator[list[Out]]:
      """Generate results as they complete."""
      futures = set()
      executor_shutdown = False

      # Pre-submit initial batch
      for chunk in itertools.islice(chunks_iter, min(self.max_workers, 10)):
        if executor_shutdown:
          break
        try:
          futures.add(executor.submit(process_chunk, chunk))
        except RuntimeError as e:
          if "cannot schedule new futures after shutdown" in str(e):
            executor_shutdown = True
            break
          raise

      while futures:
        try:
          done, futures = wait(futures, return_when=FIRST_COMPLETED)
          for future in done:
            yield future.result()

            # Try to submit next chunk only if executor is not shutdown
            if not executor_shutdown:
              try:
                chunk = next(chunks_iter)
                futures.add(executor.submit(process_chunk, chunk))
              except StopIteration:
                continue
              except RuntimeError as e:
                if "cannot schedule new futures after shutdown" in str(e):
                  executor_shutdown = True
                  continue
                raise
        except Exception:
          # Cancel remaining futures and re-raise
          for future in futures:
            try:
              future.cancel()
            except Exception:
              pass  # Ignore cancellation errors
          futures.clear()
          raise

    # Use the reusable thread pool instead of creating a new one
    executor = self._get_thread_pool(self.max_workers)
    chunks_to_process = chunk_generator(data)
    gen_func = _ordered_generator if self.ordered else _unordered_generator

    # Process chunks using the reusable executor
    for result_chunk in gen_func(chunks_to_process, executor):
      yield from result_chunk
