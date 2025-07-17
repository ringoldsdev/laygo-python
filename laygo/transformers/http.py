"""
The final, self-sufficient DistributedTransformer.
"""

from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import hashlib
import itertools
import pickle

import requests

from laygo import PipelineContext
from laygo import Transformer


class HTTPTransformer(Transformer):
  """
  A self-sufficient, chainable transformer that manages its own
  distributed execution and worker endpoint definition.
  """

  def __init__(self, base_url: str, endpoint: str | None = None, max_workers: int = 8):
    super().__init__()
    self.base_url = base_url.rstrip("/")
    self.endpoint = endpoint
    self.max_workers = max_workers
    self.session = requests.Session()
    self._worker_url: str

  def _finalize_config(self):
    """Determines the final worker URL, generating one if needed."""
    if self._worker_url:
      return

    if self.endpoint:
      path = self.endpoint
    else:
      # Using pickle to serialize the function chain and hashing for a unique ID
      serialized_logic = pickle.dumps(self.transformer)
      hash_id = hashlib.sha1(serialized_logic).hexdigest()[:16]
      path = f"/autogen/{hash_id}"

    self.endpoint = path.lstrip("/")
    self._worker_url = f"{self.base_url}/{self.endpoint}"

  def __call__(self, data: Iterable, context=None) -> Iterator:
    """CLIENT-SIDE: Called by the Pipeline to start distributed processing."""
    self._finalize_config()

    def process_chunk(chunk: list) -> list:
      """Target for a thread: sends one chunk to the worker."""
      try:
        response = self.session.post(self._worker_url, json=chunk, timeout=300)
        response.raise_for_status()
        return response.json()
      except requests.RequestException as e:
        print(f"Error calling worker {self._worker_url}: {e}")
        return []

    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
      chunk_iterator = self._chunk_generator(data)
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

  def get_route(self):
    """
    Function that returns the route for the worker.
    This is used to register the worker in a Flask app or similar.

    Returns:
        A tuple containing the endpoint and the worker function.
    """
    self._finalize_config()

    def worker_view_func(chunk: list, context: PipelineContext):
      """The actual Flask view function for this transformer's logic."""
      return self.transformer(chunk, context)

    return (f"/{self.endpoint}", worker_view_func)
