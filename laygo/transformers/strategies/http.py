from collections.abc import Callable
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
  """An execution strategy that sends data chunks to a remote HTTP worker.

  This is the CLIENT-SIDE implementation that sends chunks to a remote
  HTTP endpoint and receives the transformed results back.
  """

  def __init__(self, worker_url: Callable[[], str], max_workers: int = 8, timeout: int = 300):
    """Initialize the HTTP strategy.

    Args:
        worker_url: Function that returns the URL of the remote worker endpoint.
        max_workers: Maximum number of concurrent HTTP requests.
        timeout: Request timeout in seconds.
    """
    self.worker_url = worker_url
    self.max_workers = max_workers
    self.timeout = timeout
    self.session = requests.Session()

  def execute(
    self,
    transformer_logic: InternalTransformer[In, Out],  # Ignored for HTTP strategy
    chunks: Iterator[list[In]],
    context: IContextManager,  # Ignored for HTTP strategy
  ) -> Iterator[list[Out]]:
    """Send data chunks to the remote worker and yield results.

    Args:
        transformer_logic: Ignored - the remote worker has the transformation logic.
        chunks: Iterator of pre-chunked data.
        context: Ignored - context is handled by the remote worker.

    Returns:
        Iterator of transformed chunks received from the remote worker.
    """

    def process_chunk(chunk: list[In]) -> list[Out]:
      """Send one chunk to the worker and return the result.

      Args:
          chunk: Data chunk to send to the remote worker.

      Returns:
          Transformed chunk received from the remote worker.
      """
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
      chunk_iterator = iter(chunks)
      futures = {executor.submit(process_chunk, chunk) for chunk in itertools.islice(chunk_iterator, self.max_workers)}

      while futures:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
          yield future.result()
          try:
            new_chunk = next(chunk_iterator)
            futures.add(executor.submit(process_chunk, new_chunk))
          except StopIteration:
            continue
