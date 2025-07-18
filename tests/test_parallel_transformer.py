"""Tests for the ParallelTransformer class."""

import multiprocessing as mp
import time
from unittest.mock import patch

from laygo import ErrorHandler
from laygo import ParallelTransformer
from laygo import PipelineContext
from laygo.transformers.parallel import createParallelTransformer
from laygo.transformers.transformer import createTransformer


class TestParallelTransformerBasics:
  """Test core parallel transformer functionality."""

  def test_initialization_custom_parameters(self):
    """Test initialization with custom parameters."""
    transformer = ParallelTransformer[int, int](max_workers=8, ordered=False, chunk_size=500)
    assert transformer.max_workers == 8
    assert transformer.ordered is False
    assert transformer.chunk_size == 500

  def test_basic_execution(self):
    """Test basic parallel transformer execution."""
    transformer = ParallelTransformer[int, int](max_workers=2, chunk_size=3)
    result = list(transformer([1, 2, 3, 4, 5]))
    assert result == [1, 2, 3, 4, 5]

  def test_from_transformer_creation(self):
    """Test creating ParallelTransformer from existing Transformer."""
    regular = createTransformer(int).map(lambda x: x * 2).filter(lambda x: x > 5)
    parallel = ParallelTransformer.from_transformer(regular, max_workers=2, chunk_size=10)

    data = [1, 2, 3, 4, 5, 6]
    regular_results = list(regular(data))
    parallel_results = list(parallel(data))

    assert regular_results == parallel_results
    assert parallel.max_workers == 2
    assert parallel.ordered is True
    assert parallel.chunk_size == 10


class TestParallelTransformerOperations:
  """Test parallel transformer operations like map, filter, etc."""

  def test_map_concurrent_execution(self):
    """Test map operation with concurrent execution."""
    transformer = createParallelTransformer(int).map(lambda x: x * 2)
    result = list(transformer([1, 2, 3, 4]))
    assert result == [2, 4, 6, 8]

  def test_filter_concurrent_execution(self):
    """Test filter operation with concurrent execution."""
    transformer = createParallelTransformer(int).filter(lambda x: x % 2 == 0)
    result = list(transformer([1, 2, 3, 4, 5, 6]))
    assert result == [2, 4, 6]

  def test_chained_operations(self):
    """Test chained operations work correctly with concurrency."""
    transformer = (
      createParallelTransformer(int, chunk_size=2).map(lambda x: x * 2).filter(lambda x: x > 4).map(lambda x: x + 1)
    )
    result = list(transformer([1, 2, 3, 4, 5]))
    assert result == [7, 9, 11]

  def test_flatten_operation(self):
    """Test flatten operation with concurrent execution."""
    # This defines a transformer that accepts iterables of lists and flattens them.
    transformer = createParallelTransformer(list[int]).flatten()
    result = list(transformer([[1, 2], [3, 4], [5, 6]]))
    assert result == [1, 2, 3, 4, 5, 6]

  def test_tap_side_effects(self):
    """Test tap applies side effects correctly in concurrent execution."""
    with mp.Manager() as manager:
      side_effects = manager.list()
      transformer = createParallelTransformer(int).tap(lambda x: side_effects.append(x))
      result = list(transformer([1, 2, 3, 4]))

      assert result == [1, 2, 3, 4]
      assert sorted(side_effects) == [1, 2, 3, 4]


def safe_increment(x: int, ctx: PipelineContext) -> int:
  with ctx["lock"]:
    current_items = ctx["items"]
    time.sleep(0.001)
    ctx["items"] = current_items + 1
  return x * 2


def update_stats(x: int, ctx: PipelineContext) -> int:
  with ctx["lock"]:
    ctx["total_sum"] += x
    ctx["item_count"] += 1
    ctx["max_value"] = max(ctx["max_value"], x)
  return x * 3


class TestParallelTransformerContextSupport:
  """Test context-aware parallel transformer operations."""

  def test_map_with_context(self):
    """Test map with context-aware function in concurrent execution."""
    context = PipelineContext({"multiplier": 3})
    transformer = createParallelTransformer(int).map(lambda x, ctx: x * ctx["multiplier"])
    result = list(transformer([1, 2, 3], context))
    assert result == [3, 6, 9]

  def test_context_modification_with_locking(self):
    """Test safe context modification with locking in concurrent execution."""
    context = PipelineContext({"items": 0})

    transformer = createParallelTransformer(int, max_workers=4, chunk_size=1).map(safe_increment)
    data = list(range(1, 11))
    result = list(transformer(data, context))

    assert sorted(result) == sorted([x * 2 for x in data])
    assert context["items"] == len(data)

  def test_multiple_context_values_modification(self):
    """Test modifying multiple context values safely."""
    context = PipelineContext({"total_sum": 0, "item_count": 0, "max_value": 0})

    transformer = createParallelTransformer(int, max_workers=3, chunk_size=2).map(update_stats)
    data = [1, 5, 3, 8, 2, 7, 4, 6]
    result = list(transformer(data, context))

    assert sorted(result) == sorted([x * 3 for x in data])
    assert context["total_sum"] == sum(data)
    assert context["item_count"] == len(data)
    assert context["max_value"] == max(data)


class TestParallelTransformerOrderingAndPerformance:
  """Test ordering and performance aspects of the parallel transformer."""

  def test_ordered_execution_maintains_sequence(self):
    """Test that ordered=True maintains element order despite variable processing time."""

    def variable_time_transform(x: int) -> int:
      time.sleep(0.01 * (5 - x))  # Later elements process faster
      return x * 2

    transformer = createParallelTransformer(int, max_workers=3, ordered=True).map(variable_time_transform)
    result = list(transformer([1, 2, 3, 4, 5]))
    assert result == [2, 4, 6, 8, 10]

  def test_unordered_vs_ordered_same_elements(self):
    """Test that ordered and unordered produce same elements with different ordering."""
    data = list(range(10))
    ordered_transformer = createParallelTransformer(int, max_workers=3, ordered=True).map(lambda x: x * 2)
    ordered_result = list(ordered_transformer(data))
    unordered_transformer = createParallelTransformer(int, max_workers=3, ordered=False).map(lambda x: x * 2)
    unordered_result = list(unordered_transformer(data))

    assert sorted(ordered_result) == sorted(unordered_result)
    assert ordered_result == [x * 2 for x in data]

  def test_process_pool_management(self):
    """Test that process pool is properly created and cleaned up."""
    with patch("laygo.transformers.parallel.ProcessPoolExecutor") as mock_executor:
      mock_executor.return_value.__enter__.return_value = mock_executor.return_value
      mock_executor.return_value.__exit__.return_value = None
      mock_executor.return_value.submit.return_value.result.return_value = [2, 4]
      transformer = ParallelTransformer[int, int](max_workers=2, chunk_size=2)
      list(transformer([1, 2]))

      mock_executor.assert_called_with(max_workers=2)
      mock_executor.return_value.__enter__.assert_called_once()
      mock_executor.return_value.__exit__.assert_called_once()


class TestParallelTransformerChunkingAndEdgeCases:
  """Test chunking behavior and edge cases."""

  def test_empty_data(self):
    """Test parallel transformer with empty data."""
    transformer = createParallelTransformer(int).map(lambda x: x * 2)
    result = list(transformer([]))
    assert result == []

  def test_exception_propagation(self):
    """Test that exceptions in worker processes are properly propagated."""

    def failing_function(x: int) -> int:
      if x == 3:
        raise ValueError("Test exception")
      return x

    import pytest

    transformer = createParallelTransformer(int, chunk_size=1).map(failing_function)
    with pytest.raises(ValueError, match="Test exception"):
      list(transformer([1, 2, 3, 4]))


class TestParallelTransformerErrorHandling:
  """Test error handling with parallel transformer."""

  def test_safe_with_error_isolation(self):
    """Test safe execution isolates errors to specific chunks."""
    with mp.Manager() as manager:
      errored_chunks = manager.list()
      transformer = createParallelTransformer(int, chunk_size=1).catch(
        lambda t: t.map(lambda x: x / 0),  # Division by zero
        on_error=lambda chunk, error, context: errored_chunks.append(chunk),  # type: ignore
      )
      result = list(transformer([1, 2, 3]))
      assert result == []
      assert sorted(map(tuple, errored_chunks)) == sorted(map(tuple, [[1], [2], [3]]))

  def test_global_error_handler(self):
    """Test global error handling through error handler."""
    with mp.Manager() as manager:
      errored_chunks = manager.list()
      error_handler = ErrorHandler()
      error_handler.on_error(lambda chunk, error, context: errored_chunks.append(chunk))

      transformer = (
        createParallelTransformer(int, chunk_size=1).on_error(error_handler).catch(lambda t: t.map(lambda x: x / 0))
      )
      list(transformer([1, 2, 3]))
      assert sorted(map(tuple, errored_chunks)) == sorted(map(tuple, [[1], [2], [3]]))
