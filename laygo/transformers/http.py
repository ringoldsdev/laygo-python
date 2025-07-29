"""Distributed transformer implementation with HTTP-based worker coordination."""

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

from laygo.context import IContextManager
from laygo.context import SimpleContextManager
from laygo.errors import ErrorHandler
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
  """Create a new identity HTTP transformer with an explicit type hint.

  Args:
      _type_hint: Type hint for the data being processed.
      base_url: The base URL for the HTTP worker service.
      chunk_size: Size of chunks to process data in.
      endpoint: Optional specific endpoint path.
      max_workers: Maximum number of concurrent HTTP requests.

  Returns:
      A new identity HTTP transformer.
  """
  return HTTPTransformer[T, T](
    base_url=base_url,
    endpoint=endpoint,
    max_workers=max_workers,
    chunk_size=chunk_size,
  )


class HTTPTransformer(Transformer[In, Out]):
  """A self-sufficient, chainable transformer for distributed execution.

  This transformer manages its own distributed execution by coordinating
  with HTTP-based worker endpoints. It can automatically generate worker
  endpoints based on the transformation logic or use predefined endpoints.
  """

  def __init__(
    self,
    base_url: str,
    endpoint: str | None = None,
    max_workers: int = 8,
    chunk_size: int | None = None,
  ) -> None:
    """Initialize the HTTP transformer.

    Args:
        base_url: The base URL for the worker service.
        endpoint: Optional specific endpoint path. If not provided,
                 one will be auto-generated.
        max_workers: Maximum number of concurrent HTTP requests.
        chunk_size: Size of data chunks to process.
    """
    super().__init__(chunk_size=chunk_size)
    self.base_url = base_url.rstrip("/")
    self.endpoint = endpoint
    self.max_workers = max_workers
    self.session = requests.Session()
    self._worker_url: str | None = None
    # HTTP transformers always use a simple context manager to avoid serialization issues
    self._default_context = SimpleContextManager()

  def _finalize_config(self) -> None:
    """Determine the final worker URL, generating one if needed.

    If no explicit endpoint was provided, this method generates a unique
    endpoint based on a hash of the transformation logic.
    """
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

  def __call__(self, data: Iterable[In], context: IContextManager | None = None) -> Iterator[Out]:
    """Execute distributed processing on the data (CLIENT-SIDE).

    This method is called by the Pipeline to start distributed processing.
    It sends data chunks to worker endpoints via HTTP.

    Args:
        data: The input data to process.
        context: Optional pipeline context. HTTP transformers always use their
                internal SimpleContextManager regardless of the provided context.

    Returns:
        An iterator over the processed data.
    """
    run_context = self._default_context

    self._finalize_config()

    def process_chunk(chunk: list) -> list:
      """Send one chunk to the worker and return the result.

      Args:
          chunk: The data chunk to process.

      Returns:
          The processed chunk from the worker.
      """
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

    try:
      with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        chunk_iterator = self._chunk_generator(data)
        futures = {
          executor.submit(process_chunk, chunk) for chunk in itertools.islice(chunk_iterator, self.max_workers)
        }
        while futures:
          done, futures = wait(futures, return_when=FIRST_COMPLETED)
          for future in done:
            yield from future.result()
            try:
              new_chunk = next(chunk_iterator)
              futures.add(executor.submit(process_chunk, new_chunk))
            except StopIteration:
              continue
    finally:
      # Always clean up our context since we always use the default one
      run_context.shutdown()

  def get_route(self):
    """Get the route configuration for registering this transformer as a worker.

    This method returns the necessary information to register the worker
    in a Flask app or similar web framework.

    Returns:
        A tuple containing the endpoint path and the worker view function.
    """
    self._finalize_config()

    def worker_view_func(chunk: list, context: IContextManager):
      """The actual worker logic for this transformer.

      Args:
          chunk: The data chunk to process.
          context: The pipeline context.

      Returns:
          The processed chunk.
      """
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

  def tap(self, arg: Union["Transformer[Out, Any]", PipelineFunction[Out, Any]]) -> "HTTPTransformer[In, Out]":
    super().tap(arg)
    return self

  def apply[T](self, t: Callable[["HTTPTransformer[In, Out]"], "Transformer[In, T]"]) -> "HTTPTransformer[In, T]":
    # Note: The type hint for `t` is slightly adjusted to reflect it receives an HTTPTransformer
    super().apply(t)  # type: ignore
    return self  # type: ignore

  def catch[U](
    self,
    sub_pipeline_builder: Callable[[Transformer[Out, Out]], Transformer[Out, U]],
    on_error: ChunkErrorHandler[Out, None] | None = None,
  ) -> "HTTPTransformer[In, U]":
    super().catch(sub_pipeline_builder, on_error)
    return self  # type: ignore

  def short_circuit(self, function: Callable[[IContextManager], bool | None]) -> "HTTPTransformer[In, Out]":
    super().short_circuit(function)
    return self
