from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
import copy
from functools import reduce
import itertools
from typing import Any
from typing import Self
from typing import Union
from typing import overload

from laygo.errors import ErrorHandler
from laygo.helpers import PipelineContext
from laygo.helpers import is_context_aware
from laygo.helpers import is_context_aware_reduce

DEFAULT_CHUNK_SIZE = 1000


type PipelineFunction[Out, T] = Callable[[Out], T] | Callable[[Out, PipelineContext], T]
type PipelineReduceFunction[U, Out] = Callable[[U, Out], U] | Callable[[U, Out, PipelineContext], U]

# The internal transformer function signature is changed to explicitly accept a context.
type InternalTransformer[In, Out] = Callable[[list[In], PipelineContext], list[Out]]
type ChunkErrorHandler[In, U] = Callable[[list[In], Exception, PipelineContext], list[U]]


def createTransformer[T](_type_hint: type[T], chunk_size: int = DEFAULT_CHUNK_SIZE) -> "Transformer[T, T]":
  """Create a new identity pipeline with an explicit type hint."""
  return Transformer[T, T](chunk_size=chunk_size)  # type: ignore


def build_chunk_generator[T](chunk_size: int) -> Callable[[Iterable[T]], Iterator[list[T]]]:
  """
  Returns a function that breaks an iterable into chunks of a specified size.
  This is useful for creating transformers that process data in manageable chunks.
  """

  def chunk_generator(data: Iterable[T]) -> Iterator[list[T]]:
    data_iter = iter(data)
    while chunk := list(itertools.islice(data_iter, chunk_size)):
      yield chunk

  return chunk_generator


class Transformer[In, Out]:
  """
  Defines and composes data transformations by passing context explicitly.
  """

  def __init__(
    self,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    transformer: InternalTransformer[In, Out] | None = None,
  ):
    self.chunk_size = chunk_size
    self.context: PipelineContext = PipelineContext()
    # The default transformer now accepts and ignores a context argument.
    self.transformer: InternalTransformer[In, Out] = transformer or (lambda chunk, ctx: chunk)  # type: ignore
    self.error_handler = ErrorHandler()
    self._chunk_generator = build_chunk_generator(chunk_size) if chunk_size else lambda x: iter([list(x)])

  @classmethod
  def from_transformer[T, U](
    cls,
    transformer: "Transformer[T, U]",
    chunk_size: int | None = None,
  ) -> "Transformer[T, U]":
    """Create a new transformer from an existing one, copying its logic."""
    return cls(
      chunk_size=chunk_size or transformer.chunk_size,
      transformer=copy.deepcopy(transformer.transformer),  # type: ignore
    )

  def set_chunker(self, chunker: Callable[[Iterable[In]], Iterator[list[In]]]) -> "Transformer[In, Out]":
    """Sets a custom chunking function for the transformer."""
    self._chunk_generator = chunker
    return self

  def on_error(self, handler: ChunkErrorHandler[In, Out] | ErrorHandler) -> "Transformer[In, Out]":
    """Registers an error handler for the transformer."""
    # This method is a placeholder for future error handling logic.
    # Currently, it does not modify the transformer behavior.
    match handler:
      case ErrorHandler():
        self.error_handler = handler
      case _ if callable(handler):
        self.error_handler.on_error(handler)  # type: ignore
    return self

  def _pipe[U](self, operation: Callable[[list[Out], PipelineContext], list[U]]) -> "Transformer[In, U]":
    """Composes the current transformer with a new context-aware operation."""
    prev_transformer = self.transformer
    # The new transformer chain ensures the context `ctx` is passed at each step.
    self.transformer = lambda chunk, ctx: operation(prev_transformer(chunk, ctx), ctx)  # type: ignore
    return self  # type: ignore

  def map[U](self, function: PipelineFunction[Out, U]) -> "Transformer[In, U]":
    """Transforms elements, passing context explicitly to the mapping function."""
    if is_context_aware(function):
      return self._pipe(lambda chunk, ctx: [function(x, ctx) for x in chunk])

    return self._pipe(lambda chunk, _ctx: [function(x) for x in chunk])  # type: ignore

  def filter(self, predicate: PipelineFunction[Out, bool]) -> "Transformer[In, Out]":
    """Filters elements, passing context explicitly to the predicate function."""
    if is_context_aware(predicate):
      return self._pipe(lambda chunk, ctx: [x for x in chunk if predicate(x, ctx)])

    return self._pipe(lambda chunk, _ctx: [x for x in chunk if predicate(x)])  # type: ignore

  @overload
  def flatten[T](self: "Transformer[In, list[T]]") -> "Transformer[In, T]": ...
  @overload
  def flatten[T](self: "Transformer[In, tuple[T, ...]]") -> "Transformer[In, T]": ...
  @overload
  def flatten[T](self: "Transformer[In, set[T]]") -> "Transformer[In, T]": ...

  def flatten[T](
    self: Union["Transformer[In, list[T]]", "Transformer[In, tuple[T, ...]]", "Transformer[In, set[T]]"],
  ) -> "Transformer[In, T]":
    """Flattens nested lists; the context is passed through the operation."""
    return self._pipe(lambda chunk, ctx: [item for sublist in chunk for item in sublist])  # type: ignore

  @overload
  def tap(self, arg: "Transformer[Out, Any]") -> "Transformer[In, Out]": ...

  @overload
  def tap(self, arg: PipelineFunction[Out, Any]) -> "Transformer[In, Out]": ...

  def tap(
    self,
    arg: Union["Transformer[Out, Any]", PipelineFunction[Out, Any]],
  ) -> "Transformer[In, Out]":
    """
    Applies a side-effect without modifying the main data stream.

    This method can be used in two ways:
    1.  With a `Transformer`: Applies a sub-pipeline to each chunk for side-effects
        (e.g., logging a chunk), discarding the sub-pipeline's output.
    2.  With a `function`: Applies a function to each element individually for
        side-effects (e.g., printing an item).

    Args:
        arg: A `Transformer` instance or a function to be applied for side-effects.

    Returns:
        The transformer instance for method chaining.
    """
    match arg:
      # Case 1: The argument is another Transformer
      case Transformer() as tapped_transformer:
        tapped_func = tapped_transformer.transformer

        def operation(chunk: list[Out], ctx: PipelineContext) -> list[Out]:
          # Execute the tapped transformer's logic on the chunk for side-effects.
          _ = tapped_func(chunk, ctx)
          # Return the original chunk to continue the main pipeline.
          return chunk

        return self._pipe(operation)

      # Case 2: The argument is a callable function
      case function if callable(function):
        if is_context_aware(function):
          return self._pipe(lambda chunk, ctx: [x for x in chunk if function(x, ctx) or True])

        return self._pipe(lambda chunk, _ctx: [x for x in chunk if function(x) or True])  # type: ignore

      # Default case for robustness
      case _:
        raise TypeError(f"tap() argument must be a Transformer or a callable, not {type(arg).__name__}")

  def apply[T](self, t: Callable[[Self], "Transformer[In, T]"]) -> "Transformer[In, T]":
    """Apply another pipeline to the current one."""
    return t(self)

  def __call__(self, data: Iterable[In], context: PipelineContext | None = None) -> Iterator[Out]:
    """
    Executes the transformer on a data source.

    It uses the provided `context` by reference. If none is provided, it uses
    the transformer's internal context.
    """
    # Use the provided context by reference, or default to the instance's context.
    run_context = context or self.context

    for chunk in self._chunk_generator(data):
      # The context is now passed explicitly through the transformer chain.
      yield from self.transformer(chunk, run_context)

  def reduce[U](self, function: PipelineReduceFunction[U, Out], initial: U):
    """Reduces elements to a single value (terminal operation)."""

    if is_context_aware_reduce(function):

      def _reduce_with_context(data: Iterable[In], context: PipelineContext | None = None) -> Iterator[U]:
        # The context for the run is determined here.
        run_context = context or self.context

        data_iterator = self(data, run_context)

        def function_wrapper(acc: U, value: Out) -> U:
          return function(acc, value, run_context)

        yield reduce(function_wrapper, data_iterator, initial)

      return _reduce_with_context

    # Not context-aware, so we adapt the function to ignore the context.
    def _reduce(data: Iterable[In], context: PipelineContext | None = None) -> Iterator[U]:
      # The context for the run is determined here.
      run_context = context or self.context

      data_iterator = self(data, run_context)

      yield reduce(function, data_iterator, initial)  # type: ignore

    return _reduce

  def catch[U](
    self,
    sub_pipeline_builder: Callable[["Transformer[Out, Out]"], "Transformer[Out, U]"],
    on_error: ChunkErrorHandler[Out, U] | None = None,
  ) -> "Transformer[In, U]":
    """
    Isolates a sub-pipeline in a chunk-based try-catch block.
    If the sub-pipeline fails for a chunk, the on_error handler is invoked.
    """

    if on_error:
      self.on_error(on_error)  # type: ignore

    # Create a blank transformer for the sub-pipeline
    temp_transformer = createTransformer(_type_hint=..., chunk_size=self.chunk_size)  # type: ignore

    # Build the sub-pipeline and get its internal transformer function
    sub_pipeline = sub_pipeline_builder(temp_transformer)
    sub_transformer_func = sub_pipeline.transformer

    def operation(chunk: list[Out], ctx: PipelineContext) -> list[U]:
      try:
        # Attempt to process the whole chunk with the sub-pipeline
        return sub_transformer_func(chunk, ctx)
      except Exception as e:
        # On failure, delegate to the chunk-based error handler
        self.error_handler.handle(chunk, e, ctx)
        return []

    return self._pipe(operation)  # type: ignore

  def short_circuit(self, function: Callable[[PipelineContext], bool | None]) -> "Transformer[In, Out]":
    """
    Executes a function on the context before processing the next step for a chunk.

    This can be used for short-circuiting by raising an exception based on the
    context's state, which halts the pipeline. If the function executes
    successfully, the data chunk is passed through unmodified to the next
    operation in the chain.

    Args:
        function: A callable that accepts the `PipelineContext` as its sole
                  argument. Its return value is ignored.

    Returns:
        The transformer instance for method chaining.
    """

    def operation(chunk: list[Out], ctx: PipelineContext) -> list[Out]:
      """The internal operation that wraps the user's function."""
      # Execute the user's function with the current context.
      if function(ctx):
        # If the function returns True, we raise an exception to stop the pipeline.
        raise RuntimeError("Short-circuit condition met, stopping execution.")
      # If no exception was raised, the chunk passes through.
      return chunk

    return self._pipe(operation)
