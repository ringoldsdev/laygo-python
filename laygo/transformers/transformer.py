"""Core transformer implementation for data pipeline operations."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
import copy
from functools import reduce
import itertools
from typing import Any
from typing import Literal
from typing import Self
from typing import Union
from typing import overload

from laygo.context import IContextManager
from laygo.context import SimpleContextManager
from laygo.errors import ErrorHandler
from laygo.helpers import is_context_aware
from laygo.helpers import is_context_aware_reduce

DEFAULT_CHUNK_SIZE = 1000


type PipelineFunction[Out, T] = Callable[[Out], T] | Callable[[Out, IContextManager], T]
type PipelineReduceFunction[U, Out] = Callable[[U, Out], U] | Callable[[U, Out, IContextManager], U]

# The internal transformer function signature is changed to explicitly accept a context.
type InternalTransformer[In, Out] = Callable[[list[In], IContextManager], list[Out]]
type ChunkErrorHandler[In, U] = Callable[[list[In], Exception, IContextManager], list[U]]


def createTransformer[T](_type_hint: type[T], chunk_size: int = DEFAULT_CHUNK_SIZE) -> "Transformer[T, T]":
  """Create a new identity pipeline with an explicit type hint.

  Args:
      _type_hint: Type hint for the data being processed.
      chunk_size: Size of chunks to process data in.

  Returns:
      A new identity transformer that passes data through unchanged.
  """
  return Transformer[T, T](chunk_size=chunk_size)  # type: ignore


def build_chunk_generator[T](chunk_size: int) -> Callable[[Iterable[T]], Iterator[list[T]]]:
  """Return a function that breaks an iterable into chunks of a specified size.

  This is useful for creating transformers that process data in manageable chunks.

  Args:
      chunk_size: The size of each chunk.

  Returns:
      A function that takes an iterable and returns an iterator of chunks.
  """

  def chunk_generator(data: Iterable[T]) -> Iterator[list[T]]:
    data_iter = iter(data)
    while chunk := list(itertools.islice(data_iter, chunk_size)):
      yield chunk

  return chunk_generator


def passthrough_chunks[T](data: Iterable[list[T]]) -> Iterator[list[T]]:
  """A chunk generator that yields the entire input as a single chunk.

  This is useful for transformers that do not require chunking.

  Args:
      data: The input data to process.

  Returns:
      An iterator yielding the entire input as a single chunk.
  """
  yield from iter(data)


class Transformer[In, Out]:
  """Define and compose data transformations by passing context explicitly.

  A Transformer represents a data processing pipeline that can be chained
  together with other transformers. It supports context-aware operations,
  error handling, and chunked processing for memory efficiency.
  """

  def __init__(
    self,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    transformer: InternalTransformer[In, Out] | None = None,
  ) -> None:
    """Initialize a new transformer.

    Args:
        chunk_size: Size of chunks to process data in. If None, processes
                   all data as a single chunk.
        transformer: Optional existing transformer logic to use.
    """
    self.chunk_size = chunk_size
    # The default transformer now accepts and ignores a context argument.
    self.transformer: InternalTransformer[In, Out] = transformer or (lambda chunk, ctx: chunk)  # type: ignore
    self.error_handler = ErrorHandler()
    self._chunk_generator = build_chunk_generator(chunk_size) if chunk_size else lambda x: iter([list(x)])
    # Rule 2: Transformers create a simple context manager by default for standalone use.
    self._default_context = SimpleContextManager()

  @classmethod
  def from_transformer[T, U](
    cls,
    transformer: "Transformer[T, U]",
    chunk_size: int | None = None,
  ) -> "Transformer[T, U]":
    """Create a new transformer from an existing one, copying its logic.

    Args:
        transformer: The source transformer to copy logic from.
        chunk_size: Optional chunk size override.

    Returns:
        A new transformer with the same logic as the source.
    """
    return cls(
      chunk_size=chunk_size or transformer.chunk_size,
      transformer=copy.deepcopy(transformer.transformer),  # type: ignore
    )

  def set_chunker(self, chunker: Callable[[Iterable[In]], Iterator[list[In]]]) -> "Transformer[In, Out]":
    """Set a custom chunking function for the transformer.

    Args:
        chunker: A function that takes an iterable and returns an iterator
                of chunks.

    Returns:
        The transformer instance for method chaining.
    """
    self._chunk_generator = chunker
    return self

  def on_error(self, handler: ChunkErrorHandler[In, Out] | ErrorHandler) -> "Transformer[In, Out]":
    """Register an error handler for the transformer.

    Args:
        handler: Either an ErrorHandler instance or a chunk error handler function.

    Returns:
        The transformer instance for method chaining.
    """
    match handler:
      case ErrorHandler():
        self.error_handler = handler
      case _ if callable(handler):
        self.error_handler.on_error(handler)  # type: ignore
    return self

  def _pipe[U](self, operation: Callable[[list[Out], IContextManager], list[U]]) -> "Transformer[In, U]":
    """Compose the current transformer with a new context-aware operation.

    Args:
        operation: A function that takes a chunk and context, returning a transformed chunk.

    Returns:
        A new transformer with the composed operation.
    """
    prev_transformer = self.transformer
    # The new transformer chain ensures the context `ctx` is passed at each step.
    self.transformer = lambda chunk, ctx: operation(prev_transformer(chunk, ctx), ctx)  # type: ignore
    return self  # type: ignore

  def map[U](self, function: PipelineFunction[Out, U]) -> "Transformer[In, U]":
    """Transform elements, passing context explicitly to the mapping function.

    Args:
        function: A function to apply to each element. Can be context-aware.

    Returns:
        A new transformer with the mapping operation applied.
    """
    if is_context_aware(function):
      context_aware_func: Callable[[Out, IContextManager], U] = function  # type: ignore
      return self._pipe(lambda chunk, ctx: [context_aware_func(x, ctx) for x in chunk])

    non_context_func: Callable[[Out], U] = function  # type: ignore
    return self._pipe(lambda chunk, _ctx: [non_context_func(x) for x in chunk])

  def filter(self, predicate: PipelineFunction[Out, bool]) -> "Transformer[In, Out]":
    """Filter elements, passing context explicitly to the predicate function.

    Args:
        predicate: A function that returns True for elements to keep.
                  Can be context-aware.

    Returns:
        A transformer with the filtering operation applied.
    """
    if is_context_aware(predicate):
      context_aware_predicate: Callable[[Out, IContextManager], bool] = predicate  # type: ignore
      return self._pipe(lambda chunk, ctx: [x for x in chunk if context_aware_predicate(x, ctx)])

    non_context_predicate: Callable[[Out], bool] = predicate  # type: ignore
    return self._pipe(lambda chunk, _ctx: [x for x in chunk if non_context_predicate(x)])

  @overload
  def flatten[T](self: "Transformer[In, list[T]]") -> "Transformer[In, T]": ...
  @overload
  def flatten[T](self: "Transformer[In, tuple[T, ...]]") -> "Transformer[In, T]": ...
  @overload
  def flatten[T](self: "Transformer[In, set[T]]") -> "Transformer[In, T]": ...

  def flatten[T](
    self: Union["Transformer[In, list[T]]", "Transformer[In, tuple[T, ...]]", "Transformer[In, set[T]]"],
  ) -> "Transformer[In, T]":
    """Flatten nested collections into individual elements.

    Args:
        self: A transformer that outputs collections (list, tuple, or set).

    Returns:
        A transformer that outputs individual elements from the collections.
    """
    return self._pipe(lambda chunk, ctx: [item for sublist in chunk for item in sublist])  # type: ignore

  @overload
  def tap(self, arg: "Transformer[Out, Any]") -> "Transformer[In, Out]": ...

  @overload
  def tap(self, arg: PipelineFunction[Out, Any]) -> "Transformer[In, Out]": ...

  def tap(
    self,
    arg: Union["Transformer[Out, Any]", PipelineFunction[Out, Any]],
  ) -> "Transformer[In, Out]":
    """Apply a side-effect without modifying the main data stream.

    This method can be used in two ways:
    1. With a `Transformer`: Applies a sub-pipeline to each chunk for side-effects
       (e.g., logging a chunk), discarding the sub-pipeline's output.
    2. With a `function`: Applies a function to each element individually for
       side-effects (e.g., printing an item).

    Args:
        arg: A `Transformer` instance or a function to be applied for side-effects.

    Returns:
        The transformer instance for method chaining.

    Raises:
        TypeError: If the argument is not a Transformer or callable.
    """
    match arg:
      # Case 1: The argument is another Transformer
      case Transformer() as tapped_transformer:
        tapped_func = tapped_transformer.transformer

        def operation(chunk: list[Out], ctx: IContextManager) -> list[Out]:
          # Execute the tapped transformer's logic on the chunk for side-effects.
          _ = tapped_func(chunk, ctx)
          # Return the original chunk to continue the main pipeline.
          return chunk

        return self._pipe(operation)

      # Case 2: The argument is a callable function
      case function if callable(function):
        if is_context_aware(function):
          context_aware_func: Callable[[Out, IContextManager], Any] = function  # type: ignore
          return self._pipe(lambda chunk, ctx: [x for x in chunk if context_aware_func(x, ctx) or True])

        non_context_func: Callable[[Out], Any] = function  # type: ignore
        return self._pipe(lambda chunk, _ctx: [x for x in chunk if non_context_func(x) or True])

      # Default case for robustness
      case _:
        raise TypeError(f"tap() argument must be a Transformer or a callable, not {type(arg).__name__}")

  def apply[T](self, t: Callable[[Self], "Transformer[In, T]"]) -> "Transformer[In, T]":
    """Apply another pipeline to the current one.

    Args:
        t: A function that takes this transformer and returns a new transformer.

    Returns:
        The result of applying the function to this transformer.
    """
    return t(self)

  def loop(
    self,
    loop_transformer: "Transformer[Out, Out]",
    condition: Callable[[list[Out]], bool] | Callable[[list[Out], IContextManager], bool],
    max_iterations: int | None = None,
  ) -> "Transformer[In, Out]":
    """
    Repeatedly applies a transformer to each chunk until a condition is met.

    The loop continues as long as the `condition` function returns `True` and
    the number of iterations has not reached `max_iterations`. The provided
    `loop_transformer` must take a chunk of a certain type and return a chunk
    of the same type.

    Args:
        loop_transformer: The `Transformer` to apply in each iteration. Its
                          input and output types must match the current pipeline's
                          output type (`Transformer[Out, Out]`).
        condition: A function that takes the current chunk (and optionally
                   the `IContextManager`) and returns `True` to continue the
                   loop, or `False` to stop.
        max_iterations: An optional integer to limit the number of repetitions
                        and prevent infinite loops.

    Returns:
        The transformer instance for method chaining.
    """
    looped_func = loop_transformer.transformer
    condition_is_context_aware = is_context_aware(condition)

    def operation(chunk: list[Out], ctx: IContextManager) -> list[Out]:
      condition_checker = (  # noqa: E731
        lambda current_chunk: condition(current_chunk, ctx) if condition_is_context_aware else condition(current_chunk)  # type: ignore
      )

      current_chunk = chunk

      iterations = 0

      # The loop now uses the single `condition_checker` function.
      while (max_iterations is None or iterations < max_iterations) and condition_checker(current_chunk):  # type: ignore
        current_chunk = looped_func(current_chunk, ctx)
        iterations += 1

      return current_chunk

    return self._pipe(operation)

  def __call__(self, data: Iterable[In], context: IContextManager | None = None) -> Iterator[Out]:
    """Execute the transformer on a data source.

    It uses the provided `context` by reference. If none is provided, it uses
    the transformer's internal context.

    Args:
        data: The input data to process.
        context: Optional context (IContextManager or dict) to use during processing.

    Returns:
        An iterator over the transformed data.
    """

    # Use the provided context by reference, or default to a simple context.
    run_context = context if context is not None else self._default_context

    try:
      for chunk in self._chunk_generator(data):
        # The context is now passed explicitly through the transformer chain.
        yield from self.transformer(chunk, run_context)
    finally:
      if run_context is self._default_context:
        self._default_context.shutdown()

  @overload
  def reduce[U](
    self,
    function: PipelineReduceFunction[U, Out],
    initial: U,
    *,
    per_chunk: Literal[True],
  ) -> "Transformer[In, U]":
    """Reduces each chunk to a single value (chainable operation)."""
    ...

  @overload
  def reduce[U](
    self,
    function: PipelineReduceFunction[U, Out],
    initial: U,
    *,
    per_chunk: Literal[False] = False,
  ) -> Callable[[Iterable[In], IContextManager | None], Iterator[U]]:
    """Reduces the entire dataset to a single value (terminal operation)."""
    ...

  def reduce[U](
    self,
    function: PipelineReduceFunction[U, Out],
    initial: U,
    *,
    per_chunk: bool = False,
  ) -> Union["Transformer[In, U]", Callable[[Iterable[In], IContextManager | None], Iterator[U]]]:  # type: ignore
    """Reduces elements to a single value, either per-chunk or for the entire dataset."""
    if per_chunk:
      # --- Efficient "per-chunk" logic (chainable) ---

      # The context-awareness check is now hoisted and executed only ONCE.
      if is_context_aware_reduce(function):
        # We define a specialized operation for the context-aware case.
        context_aware_reduce_func: Callable[[U, Out, IContextManager], U] = function  # type: ignore

        def reduce_chunk_operation(chunk: list[Out], ctx: IContextManager) -> list[U]:
          if not chunk:
            return []
          # No check happens here; we know the function needs the context.
          wrapper = lambda acc, val: context_aware_reduce_func(acc, val, ctx)  # noqa: E731
          return [reduce(wrapper, chunk, initial)]
      else:
        # We define a specialized, simpler operation for the non-aware case.
        non_context_reduce_func: Callable[[U, Out], U] = function  # type: ignore

        def reduce_chunk_operation(chunk: list[Out], ctx: IContextManager) -> list[U]:
          if not chunk:
            return []
          # No check happens here; the function is called directly.
          return [reduce(non_context_reduce_func, chunk, initial)]

      return self._pipe(reduce_chunk_operation)

    # --- "Entire dataset" logic with `match` (terminal) ---
    match is_context_aware_reduce(function):
      case True:
        context_aware_reduce_func: Callable[[U, Out, IContextManager], U] = function  # type: ignore

        def _reduce_with_context(data: Iterable[In], context: IContextManager | None = None) -> Iterator[U]:
          run_context = context or self._default_context
          data_iterator = self(data, run_context)

          def function_wrapper(acc, val):
            return context_aware_reduce_func(acc, val, run_context)

          yield reduce(function_wrapper, data_iterator, initial)

        return _reduce_with_context

      case False:
        non_context_reduce_func: Callable[[U, Out], U] = function  # type: ignore

        def _reduce(data: Iterable[In], context: IContextManager | None = None) -> Iterator[U]:
          run_context = context or self._default_context
          data_iterator = self(data, run_context)
          yield reduce(non_context_reduce_func, data_iterator, initial)

        return _reduce

  def catch[U](
    self,
    sub_pipeline_builder: Callable[["Transformer[Out, Out]"], "Transformer[Out, U]"],
    on_error: ChunkErrorHandler[Out, None] | None = None,
  ) -> "Transformer[In, U]":
    """Isolate a sub-pipeline in a chunk-based try-catch block.

    If the sub-pipeline fails for a chunk, the on_error handler is invoked.

    Args:
        sub_pipeline_builder: A function that builds the sub-pipeline to protect.
        on_error: A picklable error handler for when the sub-pipeline fails.
                  It takes a chunk, exception, and context, and must return a
                  replacement chunk (`list[U]`). If not provided, an empty
                  list is returned on error.

    Returns:
        A transformer with error handling applied.
    """

    # Use the global error handler if it exists, otherwise create an internal one
    catch_error_handler = self.error_handler

    if on_error is not None:
      catch_error_handler.on_error(on_error)  # type: ignore

    # Create a blank transformer for the sub-pipeline
    temp_transformer = createTransformer(_type_hint=..., chunk_size=self.chunk_size)  # type: ignore

    # Build the sub-pipeline and get its internal transformer function
    sub_pipeline = sub_pipeline_builder(temp_transformer)
    sub_transformer_func = sub_pipeline.transformer

    # This 'operation' function is now picklable. It only closes over
    # `sub_transformer_func` and `catch_error_handler`, both of which are
    # picklable, and it no longer references `self`.
    def operation(chunk: list[Out], ctx: IContextManager) -> list[U]:
      try:
        # Attempt to process the chunk with the sub-pipeline
        return sub_transformer_func(chunk, ctx)
      except Exception as e:
        # Call the error handler (which may include both global and local handlers)
        catch_error_handler.handle(chunk, e, ctx)

        # Return an empty list as the default behavior after handling the error
        return []

    return self._pipe(operation)  # type: ignore

  def short_circuit(self, function: Callable[[IContextManager], bool | None]) -> "Transformer[In, Out]":
    """Execute a function on the context before processing the next step for a chunk.

    This can be used for short-circuiting by raising an exception based on the
    context's state, which halts the pipeline. If the function executes
    successfully, the data chunk is passed through unmodified to the next
    operation in the chain.

    Args:
        function: A callable that accepts the context (IContextManager or dict) as its sole
                  argument. If it returns True, the pipeline is stopped with
                  an exception.

    Returns:
        The transformer instance for method chaining.

    Raises:
        RuntimeError: If the function returns True, indicating a short-circuit
                     condition has been met.
    """

    def operation(chunk: list[Out], ctx: IContextManager) -> list[Out]:
      """The internal operation that wraps the user's function."""
      # Execute the user's function with the current context.
      if function(ctx):
        # If the function returns True, we raise an exception to stop the pipeline.
        raise RuntimeError("Short-circuit condition met, stopping execution.")
      # If no exception was raised, the chunk passes through.
      return chunk

    return self._pipe(operation)
