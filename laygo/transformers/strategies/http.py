from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import itertools

import requests

from laygo.context.types import IContextManager
from laygo.transformers.strategies.types import ExecutionStrategy
from laygo.transformers.types import InternalTransformer


class HTTPStrategy[In, Out](ExecutionStrategy[In, Out]):
  """
  An execution strategy that sends data chunks to a remote HTTP worker.
  This is the CLIENT-SIDE implementation.
  """

  def __init__(self, worker_url: Callable[[], str], max_workers: int = 8, timeout: int = 300):
    self.worker_url = worker_url
    self.max_workers = max_workers
    self.timeout = timeout
    self.session = requests.Session()

  def execute(
    self,
    transformer_logic: InternalTransformer[In, Out],  # Note: This is ignored
    chunk_generator: Callable[[Iterable[In]], Iterator[list[In]]],
    data: Iterable[In],
    context: IContextManager,  # Note: This is also ignored
  ) -> Iterator[Out]:
    """Sends data to the remote worker and yields results."""

    def process_chunk(chunk: list[In]) -> list[Out]:
      """Sends one chunk to the worker and returns the result."""
      try:
        response = self.session.post(
          self.worker_url(),
          json=chunk,
          timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
      except requests.RequestException as e:
        print(f"Error calling worker {self.worker_url}: {e}")
        # Depending on desired behavior, you might raise an error
        # or return an empty list to skip the failed chunk.
        return []

    # Use a ThreadPoolExecutor to make concurrent HTTP requests
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
      chunk_iterator = chunk_generator(data)
      futures = {executor.submit(process_chunk, chunk) for chunk in itertools.islice(chunk_iterator, self.max_workers)}
      while futures:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
          yield from future.result()
          try:
            new_chunk = next(chunk_iterator)
            futures.add(executor.submit(process_chunk, new_chunk))
          except StopIteration:
            continue
