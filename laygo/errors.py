from collections.abc import Callable

from laygo.helpers import PipelineContext

ChunkErrorHandler = Callable[[list, Exception, PipelineContext], None]


def raise_error(chunk: list, error: Exception, context: PipelineContext) -> None:
  """Handler that always re-raises the error, stopping execution.

  This is a default error handler that provides fail-fast behavior by
  re-raising any exception that occurs during chunk processing.

  Args:
      chunk: The data chunk that was being processed when the error occurred.
      error: The exception that was raised.
      context: The pipeline context at the time of the error.

  Raises:
      Exception: Always re-raises the provided error.
  """
  raise error


class ErrorHandler:
  """Stores and executes a chain of error handlers.

  Error handlers are executed in reverse order of addition. This design
  assumes that handlers closer to the error source should be executed first.
  """

  def __init__(self) -> None:
    """Initialize an empty error handler chain."""
    self._handlers: list[ChunkErrorHandler] = []

  def on_error(self, handler: ChunkErrorHandler) -> "ErrorHandler":
    """Add a new handler to the beginning of the chain.

    Args:
        handler: A callable that processes errors. It receives the chunk
                being processed, the exception that occurred, and the
                pipeline context.

    Returns:
        The ErrorHandler instance for method chaining.
    """
    self._handlers.insert(0, handler)
    return self

  def handle(self, chunk: list, error: Exception, context: PipelineContext) -> None:
    """Execute all handlers in the chain.

    Handlers are executed in reverse order of addition. Execution stops
    if any handler raises an exception.

    Args:
        chunk: The data chunk that was being processed when the error occurred.
        error: The exception that was raised.
        context: The pipeline context at the time of the error.
    """
    [handler(chunk, error, context) for handler in self._handlers]
