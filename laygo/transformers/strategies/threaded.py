from collections import deque
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from functools import partial
import itertools

from laygo.context.types import IContextManager
from laygo.transformers.strategies.types import ChunkGenerator
from laygo.transformers.strategies.types import ExecutionStrategy
from laygo.transformers.types import InternalTransformer


class ThreadedStrategy[In, Out](ExecutionStrategy[In, Out]):
  def __init__(self, max_workers: int = 4, ordered: bool = True):
    self.max_workers = max_workers
    self.ordered = ordered

  def execute(self, transformer_logic, chunk_generator, data, context):
    """Execute the transformer on data concurrently.

    It uses the shared context provided by the Pipeline, if available.

    Args:
        data: The input data to process.
        context: Optional pipeline context for shared state.

    Returns:
        An iterator over the transformed data.
    """

    # Since threads share memory, we can pass the context manager directly.
    # No handle/proxy mechanism is needed, but the locking inside
    # ParallelContextManager is crucial for thread safety.
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
      return transformer(chunk, shared_context)  # type: ignore

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
        chunks_to_process = chunk_generator(data)
        gen_func = _ordered_generator if self.ordered else _unordered_generator
        processed_chunks_iterator = gen_func(chunks_to_process, executor)
        for result_chunk in processed_chunks_iterator:
          yield from result_chunk

    return result_iterator_manager()
