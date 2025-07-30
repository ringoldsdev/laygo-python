"""HTTP transformer implementation for distributed data processing.

The HTTP transformer enables distributed processing by sending data chunks
to remote HTTP workers. It serves dual purposes:

1. **Server-side**: Defines transformation logic that can be exposed as an HTTP endpoint
2. **Client-side**: Sends data chunks to remote workers via HTTP requests

## Architecture

The HTTPTransformer acts as both a definition of work to be done and a client
that can distribute that work:

- When used as a **server**, it provides route configuration through `get_route()`
  that can be registered with a web framework (Flask, FastAPI, etc.)
- When used as a **client**, it automatically sends data chunks to the configured
  worker URL and collects results

## Usage Example

```python
# Create an HTTP transformer
http_transformer, get_route = create_http_transformer(
    int,
    endpoint="http://worker.example.com"
)

# Define the transformation logic
http_transformer.map(lambda x: x * 2).filter(lambda x: x > 10)

# Server-side: Get route configuration
endpoint_path, worker_func = get_route()
# Register with your web framework: app.route(endpoint_path)(worker_func)

# Client-side: Use in a pipeline
pipeline = Pipeline(data).apply(http_transformer)
results, _ = pipeline.to_list()
```

## Endpoint Resolution

The transformer handles different endpoint formats:

- **Full URL**: `"http://worker.com/process"` - Used as-is for client requests
- **Domain only**: `"http://worker.com"` - Automatically appends `/process/data`
- **Path only**: `"/custom/path"` - Used for server-side route registration
- **Auto-generated**: If no endpoint provided, generates unique path from logic hash
"""

from collections.abc import Callable
import hashlib
import pickle

from laygo.context.types import IContextManager
from laygo.transformers.strategies.http import HTTPStrategy
from laygo.transformers.transformer import DEFAULT_CHUNK_SIZE
from laygo.transformers.transformer import Transformer


def create_http_transformer[T](
  _type_hint: type[T],
  chunk_size: int = DEFAULT_CHUNK_SIZE,
  endpoint: str | None = None,
) -> tuple["HTTPTransformer[T, T]", Callable[[], tuple[str, Callable[[list[T], IContextManager], list[T]]]]]:
  """Create a new HTTP transformer with type safety and route configuration.

  This factory function creates an HTTP transformer that can be used for
  distributed processing. It returns both the transformer instance and
  a function to get the server-side route configuration.

  Args:
      _type_hint: Type hint for the data being processed. Used for type safety
                  but not functionally required.
      chunk_size: Number of items to process in each chunk. Defaults to 1000.
      endpoint: The worker endpoint specification. Can be:
                - Full URL: "http://worker.com/process" (used as-is)
                - Domain: "http://worker.com" (auto-appends path)
                - Path: "/custom/endpoint" (for server registration)
                - None: Auto-generates unique path from logic hash

  Returns:
      A tuple containing:
      - The HTTPTransformer instance for client-side use
      - A function that returns (endpoint_path, worker_function) for server setup

  Example:
      >>> transformer, get_route = create_http_transformer(
      ...     int, endpoint="http://worker.example.com"
      ... )
      >>> transformer.map(lambda x: x * 2)
      >>> endpoint, worker_func = get_route()
  """
  transformer = HTTPTransformer[T, T](chunk_size=chunk_size, endpoint=endpoint)
  return (transformer, transformer.get_route)


class HTTPTransformer[In, Out](Transformer[In, Out]):
  """A transformer that enables distributed processing via HTTP workers.

  The HTTPTransformer serves as both a client and server component for
  distributed data processing:

  - **Client mode**: Automatically sends data chunks to remote HTTP workers
  - **Server mode**: Provides route configuration for web framework integration

  The transformer uses an HTTPStrategy to handle the actual HTTP communication,
  including connection pooling, error handling, and concurrent requests.

  Attributes:
      _endpoint: The endpoint specification (URL, domain, or path)
      _final_endpoint: Cached final endpoint path for server registration

  Example:
      >>> # Create and configure transformer
      >>> transformer = HTTPTransformer(endpoint="http://worker.com")
      >>> transformer.map(lambda x: x * 2).filter(lambda x: x > 5)
      >>>
      >>> # Use as client
      >>> results = list(transformer([1, 2, 3, 4, 5]))
      >>>
      >>> # Get server configuration
      >>> path, worker_func = transformer.get_route()
  """

  def __init__(
    self,
    endpoint: str | None = None,
    chunk_size: int | None = None,
  ):
    """Initialize an HTTP transformer.

    Args:
        endpoint: The worker endpoint specification. Can be:
                  - Full URL: "http://worker.com/api/process"
                  - Domain only: "http://worker.com" (auto-appends path)
                  - Path only: "/api/process" (for server use)
                  - None: Auto-generates path from transformation hash
        chunk_size: Number of items to process per chunk. If None, uses
                    the default chunk size from the base transformer.

    Note:
        The HTTPStrategy is configured to use the worker URL determined
        by the _get_worker_url method.
    """
    super().__init__(strategy=HTTPStrategy(self._get_worker_url), chunk_size=chunk_size)
    self._endpoint = endpoint
    self._final_endpoint: str | None = None

  def _generate_endpoint_path(self) -> str:
    """Generate a unique endpoint path from the transformation logic.

    Creates a deterministic path based on the hash of the serialized
    transformation logic. This ensures that identical transformations
    will always generate the same endpoint path.

    Returns:
        A unique endpoint path in the format "/autogen/{hash}"

    Note:
        Uses SHA-1 hash of the pickled transformer function, truncated
        to 16 characters for readability.
    """
    serialized_logic = pickle.dumps(self.transformer)
    hash_id = hashlib.sha1(serialized_logic).hexdigest()[:16]
    return f"/autogen/{hash_id}"

  def _get_worker_url(self) -> str:
    """Determine the full worker URL for HTTP requests.

    Resolves the worker URL based on the endpoint configuration:

    - If no endpoint is configured, generates an auto-generated path
    - If endpoint is just a path, returns it as-is (may need base URL)
    - If endpoint is a domain-only URL, appends default "/process/data" path
    - If endpoint is a full URL with path, uses it unchanged

    Returns:
        The complete worker URL for HTTP requests.

    Examples:
        >>> transformer = HTTPTransformer(endpoint="http://worker.com")
        >>> transformer._get_worker_url()
        "http://worker.com/process/data"

        >>> transformer = HTTPTransformer(endpoint="http://worker.com/api/v1/process")
        >>> transformer._get_worker_url()
        "http://worker.com/api/v1/process"
    """
    if not self._endpoint:
      # Auto-generate endpoint path
      return self._generate_endpoint_path()
    # If endpoint is a full URL, append default path if it's just a domain

    if not self._endpoint.startswith(("http://", "https://")):
      # If it's just a path, return it (this case might need a base URL)
      return self._endpoint

    # If it already has a path beyond just '/', use as-is
    if "/" in self._endpoint.split("://", 1)[1]:
      return self._endpoint

    # Otherwise append a default path
    return f"{self._endpoint.rstrip('/')}/process/data"

  def finalize_endpoint(self) -> str:
    """Get the final endpoint path for server-side route registration.

    Extracts or generates the path component that should be used when
    registering this transformer's logic with a web framework. This
    method caches the result to ensure consistency.

    Returns:
        The endpoint path without leading slash, suitable for route registration.

    Examples:
        >>> transformer = HTTPTransformer(endpoint="http://worker.com/api/process")
        >>> transformer.finalize_endpoint()
        "api/process"

        >>> transformer = HTTPTransformer(endpoint="/custom/endpoint")
        >>> transformer.finalize_endpoint()
        "custom/endpoint"

    Note:
        The result is cached in _final_endpoint to ensure that multiple
        calls return the same value even if the transformation logic changes.
    """
    if self._final_endpoint:
      return self._final_endpoint

    if self._endpoint:
      # Extract path from full URL or use path directly
      if self._endpoint.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        parsed = urlparse(self._get_worker_url())
        path = parsed.path or "/process/data"
      else:
        path = self._endpoint
    else:
      path = self._generate_endpoint_path()

    self._final_endpoint = path.lstrip("/")
    return self._final_endpoint

  def get_route(self) -> tuple[str, Callable[[list, IContextManager], list]]:
    """Get the route configuration for web framework integration.

    Provides the endpoint path and worker function needed to register
    this transformer's logic with a web framework like Flask or FastAPI.
    The worker function executes the complete transformation pipeline
    that has been defined through chaining operations.

    Returns:
        A tuple containing:
        - Endpoint path (with leading slash) for route registration
        - Worker function that processes a chunk and returns results

    Example:
        >>> transformer = HTTPTransformer(endpoint="/api/process")
        >>> transformer.map(lambda x: x * 2).filter(lambda x: x > 5)
        >>> path, worker_func = transformer.get_route()
        >>> print(path)  # "/api/process"
        >>>
        >>> # Register with Flask
        >>> app.route(path, methods=['POST'])(worker_func)
        >>>
        >>> # Or with FastAPI
        >>> app.post(path)(worker_func)

    Note:
        The worker function signature matches the expected format for
        HTTP endpoints: it takes a list (the JSON payload) and a context
        manager, returning a list of processed results.
    """
    endpoint = self.finalize_endpoint()

    def worker_view_func(chunk: list, context: IContextManager) -> list:
      """Execute the transformation logic on a data chunk.

      This function represents the actual worker logic that processes
      incoming data chunks. It applies all the transformations that
      have been chained onto this HTTPTransformer instance.

      Args:
          chunk: List of data items to process (from HTTP request JSON).
          context: Context manager for sharing state during processing.

      Returns:
          List of processed results (will be returned as HTTP response JSON).
      """
      # The `self.transformer` holds the composed function (e.g., map -> filter)
      return self.transformer(chunk, context)

    return (f"/{endpoint}", worker_view_func)
