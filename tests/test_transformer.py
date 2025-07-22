"""Tests for the Transformer class."""

import pytest

from laygo import ErrorHandler
from laygo import PipelineContext
from laygo import Transformer
from laygo.transformers.transformer import createTransformer


class TestTransformerBasics:
  """Test core transformer functionality."""

  def test_identity_transformer(self):
    """Test that init creates an identity transformer."""
    transformer = createTransformer(int)
    result = list(transformer([1, 2, 3]))
    assert result == [1, 2, 3]

  def test_custom_chunk_size(self):
    """Test transformer with custom chunk size."""
    transformer = createTransformer(int, chunk_size=2)
    assert transformer.chunk_size == 2
    # Functionality should work regardless of chunk size
    result = list(transformer([1, 2, 3, 4]))
    assert result == [1, 2, 3, 4]


class TestTransformerOperations:
  """Test individual transformer operations."""

  def test_map_transformation(self):
    """Test map transforms each element."""
    transformer = createTransformer(int).map(lambda x: x * 2)
    result = list(transformer([1, 2, 3]))
    assert result == [2, 4, 6]

  def test_filter_operation(self):
    """Test filter keeps only matching elements."""
    transformer = createTransformer(int).filter(lambda x: x % 2 == 0)
    result = list(transformer([1, 2, 3, 4, 5, 6]))
    assert result == [2, 4, 6]

  def test_flatten_operation(self):
    """Test flatten with various iterable types."""
    # Test with lists
    transformer = createTransformer(list).flatten()
    result = list(transformer([[1, 2], [3, 4], [5]]))
    assert result == [1, 2, 3, 4, 5]

  def test_tap_side_effects(self):
    """Test tap applies side effects without modifying data."""
    side_effects = []
    transformer = createTransformer(int).tap(lambda x: side_effects.append(x))
    result = list(transformer([1, 2, 3]))

    assert result == [1, 2, 3]  # Data unchanged
    assert side_effects == [1, 2, 3]  # Side effect applied

  def test_loop_basic_operation(self):
    """Test loop applies transformer repeatedly until condition is met."""
    # Create a loop transformer that adds 1 to each element
    increment_transformer = createTransformer(int).map(lambda x: x + 1)

    # Continue looping while any element is less than 5
    def condition(chunk):
      return any(x < 5 for x in chunk)

    transformer = createTransformer(int).loop(increment_transformer, condition, max_iterations=10)
    result = list(transformer([1, 2, 3]))

    # Should increment until all elements are >= 5: [1,2,3] -> [2,3,4] -> [3,4,5] -> [4,5,6] -> [5,6,7]
    assert result == [5, 6, 7]

  def test_loop_with_max_iterations(self):
    """Test loop respects max_iterations limit."""
    # Create a loop transformer that adds 1 to each element
    increment_transformer = createTransformer(int).map(lambda x: x + 1)

    # Condition that would normally continue indefinitely
    def always_true_condition(chunk):
      return True

    transformer = createTransformer(int).loop(increment_transformer, always_true_condition, max_iterations=3)
    result = list(transformer([1, 2, 3]))

    # Should stop after 3 iterations: [1,2,3] -> [2,3,4] -> [3,4,5] -> [4,5,6]
    assert result == [4, 5, 6]

  def test_loop_no_iterations(self):
    """Test loop when condition is false from the start."""
    increment_transformer = createTransformer(int).map(lambda x: x + 1)

    # Condition that's immediately false
    def exit_immediately(chunk):
      return False

    transformer = createTransformer(int).loop(increment_transformer, exit_immediately)
    result = list(transformer([1, 2, 3]))

    # Should not iterate at all
    assert result == [1, 2, 3]


class TestTransformerContextSupport:
  """Test context-aware transformer operations."""

  def test_map_with_context(self):
    """Test map with context-aware function."""
    context = PipelineContext({"multiplier": 3})
    transformer = Transformer().map(lambda x, ctx: x * ctx["multiplier"])
    result = list(transformer([1, 2, 3], context))
    assert result == [3, 6, 9]

  def test_filter_with_context(self):
    """Test filter with context-aware function."""
    context = PipelineContext({"threshold": 3})
    transformer = Transformer().filter(lambda x, ctx: x > ctx["threshold"])
    result = list(transformer([1, 2, 3, 4, 5], context))
    assert result == [4, 5]

  def test_tap_with_context(self):
    """Test tap with context-aware function."""
    side_effects = []
    context = PipelineContext({"prefix": "item:"})
    transformer = Transformer().tap(lambda x, ctx: side_effects.append(f"{ctx['prefix']}{x}"))
    result = list(transformer([1, 2, 3], context))

    assert result == [1, 2, 3]
    assert side_effects == ["item:1", "item:2", "item:3"]

  def test_tap_with_transformer(self):
    """Test tap with a transformer for side effects."""
    side_effects = []

    # Create a side-effect transformer that logs processed values
    side_effect_transformer = (
      createTransformer(int)
      .map(lambda x: x * 10)  # Transform for side effect
      .tap(lambda x: side_effects.append(x))  # Capture the transformed values
    )

    # Main transformer that uses the side-effect transformer via tap
    main_transformer = (
      createTransformer(int)
      .map(lambda x: x * 2)  # Main transformation
      .tap(side_effect_transformer)  # Apply side-effect transformer
      .map(lambda x: x + 1)  # Continue main transformation
    )

    result = list(main_transformer([1, 2, 3]))

    # Main pipeline should produce: [1,2,3] -> [2,4,6] -> [3,5,7]
    assert result == [3, 5, 7]

    # Side effects should capture: [2,4,6] -> [20,40,60]
    assert side_effects == [20, 40, 60]

  def test_tap_with_transformer_and_context(self):
    """Test tap with a transformer that uses context."""
    side_effects = []
    context = PipelineContext({"multiplier": 5, "log_prefix": "processed:"})

    # Create a context-aware side-effect transformer
    side_effect_transformer = (
      createTransformer(int)
      .map(lambda x, ctx: x * ctx["multiplier"])  # Use context multiplier
      .tap(lambda x, ctx: side_effects.append(f"{ctx['log_prefix']}{x}"))  # Log with context prefix
    )

    # Main transformer
    main_transformer = (
      createTransformer(int)
      .map(lambda x: x + 10)  # Main transformation
      .tap(side_effect_transformer)  # Apply side-effect transformer with context
    )

    result = list(main_transformer([1, 2, 3], context))

    # Main pipeline: [1,2,3] -> [11,12,13]
    assert result == [11, 12, 13]

    # Side effects: [11,12,13] -> [55,60,65] -> ["processed:55", "processed:60", "processed:65"]
    assert side_effects == ["processed:55", "processed:60", "processed:65"]

  def test_loop_with_context(self):
    """Test loop with context-aware condition and transformer."""
    side_effects = []
    context = PipelineContext({"target_sum": 15, "increment": 2})

    # Create a context-aware loop transformer that uses context increment
    loop_transformer = (
      createTransformer(int)
      .map(lambda x, ctx: x + ctx["increment"])  # Use context increment
      .tap(lambda x, ctx: side_effects.append(f"iteration:{x}"))  # Log each iteration
    )

    # Context-aware condition: continue while sum of chunk is less than target_sum
    def condition_with_context(chunk, ctx):
      return sum(chunk) < ctx["target_sum"]

    main_transformer = createTransformer(int).loop(loop_transformer, condition_with_context, max_iterations=10)

    result = list(main_transformer([1, 2, 3], context))

    # Initial: [1,2,3] sum=6 < 15, continue
    # After 1st: [3,4,5] sum=12 < 15, continue
    # After 2nd: [5,6,7] sum=18 >= 15, stop
    assert result == [5, 6, 7]

    # Should have logged both iterations
    assert side_effects == ["iteration:3", "iteration:4", "iteration:5", "iteration:5", "iteration:6", "iteration:7"]

  def test_loop_with_context_and_side_effects(self):
    """Test loop with context-aware condition that reads context data."""
    context = PipelineContext({"max_value": 20, "increment": 3})

    # Simple loop transformer that uses context increment
    loop_transformer = createTransformer(int).map(lambda x, ctx: x + ctx["increment"])

    # Context-aware condition: continue while max value in chunk is less than context max_value
    def condition_with_context(chunk, ctx):
      return max(chunk) < ctx["max_value"]

    main_transformer = createTransformer(int).loop(loop_transformer, condition_with_context, max_iterations=10)

    result = list(main_transformer([5, 8], context))

    # [5,8] -> [8,11] -> [11,14] -> [14,17] -> [17,20] (stop because max(17,20) >= 20)
    assert result == [17, 20]


class TestTransformerChaining:
  """Test chaining multiple transformer operations."""

  def test_map_filter_chain(self):
    """Test map followed by filter."""
    transformer = createTransformer(int).map(lambda x: x * 2).filter(lambda x: x > 4)
    result = list(transformer([1, 2, 3, 4]))
    assert result == [6, 8]

  def test_complex_operation_chain(self):
    """Test complex chain with multiple operations."""
    transformer = (
      createTransformer(int)
      .map(lambda x: [x, x * 2])  # Create pairs
      .flatten()  # Flatten to single list
      .filter(lambda x: x > 3)  # Keep values > 3
    )
    result = list(transformer([1, 2, 3]))
    assert result == [4, 6]  # [[1,2], [2,4], [3,6]] -> [1,2,2,4,3,6] -> [4,6]

  def test_transformer_composition(self):
    """Test transformer composition with apply."""
    base_transformer = createTransformer(int).map(lambda x: x * 2)
    composed_transformer = base_transformer.apply(lambda t: t.filter(lambda x: x > 4))
    result = list(composed_transformer([1, 2, 3, 4]))
    assert result == [6, 8]


class TestTransformerReduceOperations:
  """Test terminal reduce operations."""

  def test_basic_reduce(self):
    """Test reduce with sum operation."""
    transformer = createTransformer(int)
    reducer = transformer.reduce(lambda acc, x: acc + x, initial=0)
    result = list(reducer([1, 2, 3, 4]))
    assert result == [10]

  def test_reduce_with_context(self):
    """Test reduce with context-aware function."""
    context = PipelineContext({"multiplier": 2})
    transformer = Transformer()
    reducer = transformer.reduce(lambda acc, x, ctx: acc + (x * ctx["multiplier"]), initial=0)
    result = list(reducer([1, 2, 3], context))
    assert result == [12]  # (1*2) + (2*2) + (3*2) = 12

  def test_reduce_after_transformation(self):
    """Test reduce after map transformation."""
    transformer = createTransformer(int).map(lambda x: x * 2)
    reducer = transformer.reduce(lambda acc, x: acc + x, initial=0)
    result = list(reducer([1, 2, 3]))
    assert result == [12]  # [2, 4, 6] summed = 12


class TestTransformerEdgeCases:
  """Test edge cases and boundary conditions."""

  def test_empty_data(self):
    """Test transformer with empty data."""
    transformer = createTransformer(int).map(lambda x: x * 2)
    result = list(transformer([]))
    assert result == []

  def test_single_element(self):
    """Test transformer with single element."""
    transformer = createTransformer(int).map(lambda x: x * 2).filter(lambda x: x > 0)
    result = list(transformer([5]))
    assert result == [10]

  def test_filter_removes_all_elements(self):
    """Test filter that removes all elements."""
    transformer = createTransformer(int).filter(lambda x: x > 100)
    result = list(transformer([1, 2, 3]))
    assert result == []

  def test_chunking_behavior(self):
    """Test that chunking doesn't affect final results."""
    data = list(range(100))

    # Small chunks
    small_chunk_transformer = createTransformer(int, chunk_size=5).map(lambda x: x * 2)
    small_result = list(small_chunk_transformer(data))

    # Large chunks
    large_chunk_transformer = createTransformer(int, chunk_size=50).map(lambda x: x * 2)
    large_result = list(large_chunk_transformer(data))

    # Results should be identical regardless of chunk size
    assert small_result == large_result


class TestTransformerFromTransformer:
  """Test transformer copying and creation from existing transformers."""

  def test_copy_transformer_logic(self):
    """Test that from_transformer copies transformation logic."""
    source = createTransformer(int, chunk_size=50).map(lambda x: x * 3).filter(lambda x: x > 6)
    target = Transformer.from_transformer(source)

    data = [1, 2, 3, 4, 5]
    source_result = list(source(data))
    target_result = list(target(data))

    assert source_result == target_result
    assert target.chunk_size == 50

  def test_copy_with_custom_parameters(self):
    """Test from_transformer with custom parameters."""
    source = createTransformer(int).map(lambda x: x * 2)
    target = Transformer.from_transformer(source, chunk_size=200)

    assert target.chunk_size == 200
    # Should still have same transformation logic
    data = [1, 2, 3]
    assert list(source(data)) == list(target(data))


class TestTransformerErrorHandling:
  """Test error handling and safe operations."""

  def test_catch_with_successful_operation(self):
    """Test catch with successful transformation."""
    transformer = createTransformer(int).catch(lambda t: t.map(lambda x: x * 2))
    result = list(transformer([1, 2, 3]))
    assert result == [2, 4, 6]

  def test_catch_with_error_isolation(self):
    """Test catch isolates errors to specific chunks."""
    errored_chunks = []
    transformer = createTransformer(int, chunk_size=1).catch(
      lambda t: t.map(lambda x: x / 0),  # Division by zero
      on_error=lambda chunk, error, context: errored_chunks.append(chunk),  # type: ignore
    )
    result = list(transformer([1, 2, 3]))

    assert result == []  # All operations failed
    assert errored_chunks == [[1], [2], [3]]  # Each chunk failed individually

  def test_global_error_handler(self):
    """Test global error handling through error handler."""
    errored_chunks = []
    error_handler = ErrorHandler()
    error_handler.on_error(lambda chunk, error, context: errored_chunks.append(chunk))

    transformer = createTransformer(int, chunk_size=1).on_error(error_handler).catch(lambda t: t.map(lambda x: x / 0))

    list(transformer([1, 2, 3]))
    assert errored_chunks == [[1], [2], [3]]

  def test_short_circuit_on_error(self):
    """Test short-circuit behavior when errors occur."""

    def set_error_flag(_chunk, _error, context):
      context["error_occurred"] = True

    transformer = (
      createTransformer(int, chunk_size=1)
      .catch(
        lambda t: t.map(lambda x: x / 0),
        on_error=set_error_flag,  # type: ignore
      )
      .short_circuit(lambda ctx: ctx.get("error_occurred", False))
    )

    with pytest.raises(RuntimeError):
      list(transformer([1, 2, 3]))

  def test_short_circuit_with_custom_exception(self):
    """Test short-circuit with custom exception raising."""

    def set_error_flag(_chunk, _error, context):
      context["error_occurred"] = True

    def raise_on_error(ctx):
      if ctx.get("error_occurred"):
        raise RuntimeError("Short-circuit condition met, stopping execution.")

    transformer = (
      createTransformer(int, chunk_size=1)
      .catch(
        lambda t: t.map(lambda x: x / 0),
        on_error=set_error_flag,  # type: ignore
      )
      .short_circuit(raise_on_error)
    )

    with pytest.raises(RuntimeError, match="Short-circuit condition met"):
      list(transformer([1, 2, 3]))
