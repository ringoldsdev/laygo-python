"""
The final, self-sufficient DistributedTransformer with corrected typing.
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import hashlib
import itertools
import pickle
from typing import Any
from typing import TypeVar
from typing import Union
from typing import overload

import requests

from laygo.errors import ErrorHandler
from laygo.helpers import PipelineContext
from laygo.transformers.transformer import ChunkErrorHandler
from laygo.transformers.transformer import PipelineFunction
from laygo.transformers.transformer import Transformer

In = TypeVar("In")
Out = TypeVar("Out")
T = TypeVar("T")
U = TypeVar("U")


def createHTTPTransformer[T](
  _type_hint: type[T],
  base_url: str,
  chunk_size: int | None = None,
  endpoint: str | None = None,
  max_workers: int = 4,
) -> "HTTPTransformer[T, T]":
  """Create a new identity parallel transformer with an explicit type hint."""
  return HTTPTransformer[T, T](
    base_url=base_url,
    endpoint=endpoint,
    max_workers=max_workers,
    chunk_size=chunk_size,
  )


class HTTPTransformer(Transformer[In, Out]):
  """
  A self-sufficient, chainable transformer that manages its own
  distributed execution and worker endpoint definition.
  """

  def __init__(self, base_url: str, endpoint: str | None = None, max_workers: int = 8, chunk_size: int | None = None):
    super().__init__(chunk_size=chunk_size)
    self.base_url = base_url.rstrip("/")
    self.endpoint = endpoint
    self.max_workers = max_workers
    self.session = requests.Session()
    self._worker_url: str | None = None

  def _finalize_config(self):
    """Determines the final worker URL, generating one if needed."""
    if hasattr(self, "_worker_url") and self._worker_url:
      return

    if self.endpoint:
      path = self.endpoint
    else:
      if not self.transformer:
        raise ValueError("Cannot determine endpoint for an empty transformer.")
      serialized_logic = pickle.dumps(self.transformer)
      hash_id = hashlib.sha1(serialized_logic).hexdigest()[:16]
      path = f"/autogen/{hash_id}"

    self.endpoint = path.lstrip("/")
    self._worker_url = f"{self.base_url}/{self.endpoint}"

  # --- Original HTTPTransformer Methods ---

  def __call__(self, data: Iterable[In], context: PipelineContext | None = None) -> Iterator[Out]:
    """CLIENT-SIDE: Called by the Pipeline to start distributed processing."""
    self._finalize_config()

    def process_chunk(chunk: list) -> list:
      """Target for a thread: sends one chunk to the worker."""
      try:
        response = self.session.post(
          self._worker_url,  # type: ignore
          json=chunk,
          timeout=300,
        )
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
    """
    self._finalize_config()

    def worker_view_func(chunk: list, context: PipelineContext):
      """The actual worker logic for this transformer."""
      return self.transformer(chunk, context)

    return (f"/{self.endpoint}", worker_view_func)

  # --- Overridden Chaining Methods to Preserve Type ---

  def on_error(self, handler: ChunkErrorHandler[In, Out] | ErrorHandler) -> "HTTPTransformer[In, Out]":
    super().on_error(handler)
    return self

  def map[U](self, function: PipelineFunction[Out, U]) -> "HTTPTransformer[In, U]":
    super().map(function)
    return self  # type: ignore

  def filter(self, predicate: PipelineFunction[Out, bool]) -> "HTTPTransformer[In, Out]":
    super().filter(predicate)
    return self

  @overload
  def flatten[T](self: "HTTPTransformer[In, list[T]]") -> "HTTPTransformer[In, T]": ...
  @overload
  def flatten[T](self: "HTTPTransformer[In, tuple[T, ...]]") -> "HTTPTransformer[In, T]": ...
  @overload
  def flatten[T](self: "HTTPTransformer[In, set[T]]") -> "HTTPTransformer[In, T]": ...
  # Forgive me for I have sinned, but this is necessary to avoid type errors
  # Sinec I'm setting self type in the parent class, overriding it isn't allowed
  def flatten[T](  # type: ignore
    self: Union["HTTPTransformer[In, list[T]]", "HTTPTransformer[In, tuple[T, ...]]", "HTTPTransformer[In, set[T]]"],
  ) -> "HTTPTransformer[In, T]":
    super().flatten()  # type: ignore
    return self  # type: ignore

  def tap(self, function: PipelineFunction[Out, Any]) -> "HTTPTransformer[In, Out]":
    super().tap(function)
    return self

  def apply[T](self, t: Callable[["HTTPTransformer[In, Out]"], "Transformer[In, T]"]) -> "HTTPTransformer[In, T]":
    # Note: The type hint for `t` is slightly adjusted to reflect it receives an HTTPTransformer
    super().apply(t)  # type: ignore
    return self  # type: ignore

  def catch[U](
    self,
    sub_pipeline_builder: Callable[[Transformer[Out, Out]], Transformer[Out, U]],
    on_error: ChunkErrorHandler[Out, U] | None = None,
  ) -> "HTTPTransformer[In, U]":
    super().catch(sub_pipeline_builder, on_error)
    return self  # type: ignore

  def short_circuit(self, function: Callable[[PipelineContext], bool | None]) -> "HTTPTransformer[In, Out]":
    super().short_circuit(function)
    return self
