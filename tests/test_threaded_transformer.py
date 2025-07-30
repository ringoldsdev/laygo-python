"""Tests for the ThreadedTransformer class."""

import time

from laygo import ErrorHandler
from laygo.context.parallel import ParallelContextManager
from laygo.context.types import IContextManager
from laygo.transformers.transformer import Transformer
from laygo.transformers.transformer import create_threaded_transformer
from laygo.transformers.transformer import create_transformer


class TestThreadedTransformerBasics:
  """Test core parallel transformer functionality."""

  def test_basic_execution(self):
    """Test basic parallel transformer execution."""
    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=3)
    result = list(transformer([1, 2, 3, 4, 5]))
    assert result == [1, 2, 3, 4, 5]

  def test_from_transformer_creation(self):
    """Test creating ThreadedTransformer from existing Transformer."""
    regular = create_transformer(int, chunk_size=100).map(lambda x: x * 2).filter(lambda x: x > 5)
    parallel = Transformer.from_transformer(regular)

    data = [1, 2, 3, 4, 5, 6]
    regular_results = list(regular(data))
    parallel_results = list(parallel(data))

    assert regular_results == parallel_results


class TestThreadedTransformerOperations:
  """Test parallel transformer operations like map, filter, etc."""

  def test_map_concurrent_execution(self):
    """Test map operation with concurrent execution."""
    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=2).map(lambda x: x * 2)
    result = list(transformer([1, 2, 3, 4]))
    assert result == [2, 4, 6, 8]

  def test_filter_concurrent_execution(self):
    """Test filter operation with concurrent execution."""
    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=2).filter(lambda x: x % 2 == 0)
    result = list(transformer([1, 2, 3, 4, 5, 6]))
    assert result == [2, 4, 6]

  def test_chained_operations(self):
    """Test chained operations work correctly with concurrency."""
    transformer = (
      create_threaded_transformer(int, max_workers=2, chunk_size=2)
      .map(lambda x: x * 2)
      .filter(lambda x: x > 4)
      .map(lambda x: x + 1)
    )
    result = list(transformer([1, 2, 3, 4, 5]))
    assert result == [7, 9, 11]  # [2,4,6,8,10] -> [6,8,10] -> [7,9,11]

  def test_flatten_operation(self):
    """Test flatten operation with concurrent execution."""
    transformer = create_threaded_transformer(list[int], max_workers=2, chunk_size=2).flatten()
    result = list(transformer([[1, 2], [3, 4], [5, 6]]))
    assert result == [1, 2, 3, 4, 5, 6]

  def test_tap_side_effects(self):
    """Test tap applies side effects correctly in concurrent execution."""
    side_effects = []
    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=2)
    transformer = transformer.tap(lambda x: side_effects.append(x))
    result = list(transformer([1, 2, 3, 4]))

    assert result == [1, 2, 3, 4]  # Data unchanged
    assert sorted(side_effects) == [1, 2, 3, 4]  # Side effects applied (may be out of order)


class TestThreadedTransformerContextSupport:
  """Test context-aware parallel transformer operations."""

  def test_map_with_context(self):
    """Test map with context-aware function in concurrent execution."""
    context = ParallelContextManager({"multiplier": 3})
    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=2)
    transformer = transformer.map(lambda x, ctx: x * ctx["multiplier"])
    result = list(transformer([1, 2, 3], context))
    assert result == [3, 6, 9]

  def test_context_modification_with_locking(self):
    """Test safe context modification with locking in concurrent execution."""
    context = ParallelContextManager({"items": 0})

    def safe_increment(x: int, ctx: IContextManager) -> int:
      with ctx:
        # Simulate a race condition
        time.sleep(0.001)  # Increase chance of race condition
        ctx["items"] = ctx["items"] + 1
      return x * 2

    transformer = create_threaded_transformer(int, max_workers=4, chunk_size=1)
    transformer = transformer.map(safe_increment)

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = list(transformer(data, context))

    assert sorted(result) == sorted([x * 2 for x in data])
    assert context["items"] == len(data)

  def test_multiple_context_values_modification(self):
    """Test modifying multiple context values safely."""
    context = ParallelContextManager({"total_sum": 0, "item_count": 0, "max_value": 0})

    def update_stats(x: int, ctx: IContextManager) -> int:
      with ctx:
        ctx["total_sum"] += x
        ctx["item_count"] += 1
        ctx["max_value"] = max(ctx["max_value"], x)
      return x * 3

    transformer = create_threaded_transformer(int, max_workers=3, chunk_size=2)
    transformer = transformer.map(update_stats)

    data = [1, 5, 3, 8, 2, 7, 4, 6]
    result = list(transformer(data, context))

    assert sorted(result) == sorted([x * 3 for x in data])
    assert context["total_sum"] == sum(data)
    assert context["item_count"] == len(data)
    assert context["max_value"] == max(data)


class TestThreadedTransformerOrdering:
  """Test ordering behavior of parallel transformer."""

  def test_ordered_execution_maintains_sequence(self):
    """Test that ordered=True maintains element order despite variable processing time."""

    def variable_time_transform(x: int) -> int:
      time.sleep(0.01 * (5 - x))  # Later elements process faster
      return x * 2

    transformer = create_threaded_transformer(int, max_workers=3, ordered=True, chunk_size=2)
    transformer = transformer.map(variable_time_transform)
    result = list(transformer([1, 2, 3, 4, 5]))

    assert result == [2, 4, 6, 8, 10]  # Order maintained

  def test_unordered_vs_ordered_same_elements(self):
    """Test that ordered and unordered produce same elements with different ordering."""
    data = list(range(10))

    ordered_transformer = create_threaded_transformer(int, max_workers=3, ordered=True, chunk_size=3)
    ordered_result = list(ordered_transformer.map(lambda x: x * 2)(data))

    unordered_transformer = create_threaded_transformer(int, max_workers=3, ordered=False, chunk_size=3)
    unordered_result = list(unordered_transformer.map(lambda x: x * 2)(data))

    assert sorted(ordered_result) == sorted(unordered_result)
    assert ordered_result == [x * 2 for x in data]  # Ordered maintains sequence


class TestThreadedTransformerChunking:
  """Test chunking behavior with concurrent execution."""

  def test_chunking_effectiveness(self):
    """Test that chunking works correctly with concurrent execution."""
    processed_chunks = []

    def track_processing(x: int) -> int:
      processed_chunks.append(x)
      return x * 2

    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=3)
    transformer = transformer.map(track_processing)
    result = list(transformer([1, 2, 3, 4, 5, 6, 7]))

    assert result == [2, 4, 6, 8, 10, 12, 14]
    assert sorted(processed_chunks) == [1, 2, 3, 4, 5, 6, 7]

  def test_large_chunk_size_handling(self):
    """Test parallel transformer with large chunk size relative to data."""
    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=1000)
    transformer = transformer.map(lambda x: x + 1)
    large_data = list(range(100))  # Much smaller than chunk size
    result = list(transformer(large_data))
    expected = [x + 1 for x in large_data]
    assert result == expected


class TestThreadedTransformerEdgeCases:
  """Test edge cases and boundary conditions."""

  def test_empty_data(self):
    """Test parallel transformer with empty data."""
    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=2).map(lambda x: x * 2)
    result = list(transformer([]))
    assert result == []

  def test_single_element(self):
    """Test parallel transformer with single element."""
    transformer = (
      create_threaded_transformer(int, max_workers=2, chunk_size=2).map(lambda x: x * 2).filter(lambda x: x > 0)
    )
    result = list(transformer([5]))
    assert result == [10]

  def test_data_smaller_than_chunk_size(self):
    """Test when data is smaller than chunk size."""
    transformer = create_threaded_transformer(int, max_workers=4, chunk_size=100)
    transformer = transformer.map(lambda x: x * 2)
    result = list(transformer([1, 2, 3]))
    assert result == [2, 4, 6]

  def test_more_workers_than_chunks(self):
    """Test when workers exceed number of chunks."""
    transformer = create_threaded_transformer(int, max_workers=10, chunk_size=2)
    transformer = transformer.map(lambda x: x * 2)
    result = list(transformer([1, 2, 3]))  # Only 2 chunks, but 10 workers
    assert result == [2, 4, 6]

  def test_exception_propagation(self):
    """Test that exceptions in worker threads are properly propagated."""

    def failing_function(x: int) -> int:
      if x == 3:
        raise ValueError("Test exception")
      return x * 2

    transformer = create_threaded_transformer(int, max_workers=2, chunk_size=2)
    transformer = transformer.map(failing_function)

    try:
      list(transformer([1, 2, 3, 4]))
      raise AssertionError("Expected exception was not raised")
    except ValueError as e:
      assert "Test exception" in str(e)


class TestThreadedTransformerErrorHandling:
  """Test error handling with parallel transformer."""

  def test_safe_with_successful_operation(self):
    """Test safe execution with successful transformation."""
    transformer = create_threaded_transformer(int).catch(lambda t: t.map(lambda x: x * 2))
    result = list(transformer([1, 2, 3]))
    assert result == [2, 4, 6]

  def test_safe_with_error_isolation(self):
    """Test safe execution isolates errors to specific chunks."""
    errored_chunks = []
    transformer = create_threaded_transformer(int, chunk_size=1).catch(
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

    transformer = (
      create_threaded_transformer(int, chunk_size=1).on_error(error_handler).catch(lambda t: t.map(lambda x: x / 0))
    )

    list(transformer([1, 2, 3]))
    assert errored_chunks == [[1], [2], [3]]
