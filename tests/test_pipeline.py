"""Tests for the Pipeline class."""

from laygo import Pipeline
from laygo import PipelineContext
from laygo.transformers.transformer import createTransformer


class TestPipelineBasics:
  """Test core pipeline functionality."""

  def test_single_iterable_creation(self):
    """Test creating pipeline from single iterable."""
    pipeline = Pipeline([1, 2, 3])
    assert pipeline.to_list() == [1, 2, 3]

  def test_multiple_iterables_creation(self):
    """Test creating pipeline from multiple iterables."""
    pipeline = Pipeline([1, 2], [3, 4], [5])
    assert pipeline.to_list() == [1, 2, 3, 4, 5]

  def test_pipeline_iteration(self):
    """Test pipeline is iterable."""
    pipeline = Pipeline([1, 2, 3])
    assert list(pipeline) == [1, 2, 3]

  def test_iterator_consumption(self):
    """Test that to_list consumes the iterator."""
    pipeline = Pipeline([1, 2, 3])
    first_result = pipeline.to_list()
    second_result = pipeline.to_list()
    assert first_result == [1, 2, 3]
    assert second_result == []  # Iterator is consumed


class TestPipelineTransformations:
  """Test pipeline transformation methods."""

  def test_apply_with_transformer(self):
    """Test apply with transformer object."""
    transformer = createTransformer(int).map(lambda x: x * 2).filter(lambda x: x > 4)
    result = Pipeline([1, 2, 3, 4]).apply(transformer).to_list()
    assert result == [6, 8]

  def test_apply_with_generator_function(self):
    """Test apply with generator function."""

    def double_generator(data):
      for item in data:
        yield item * 2

    result = Pipeline([1, 2, 3]).apply(double_generator).to_list()
    assert result == [2, 4, 6]

  def test_transform_shorthand(self):
    """Test transform shorthand method."""
    result = Pipeline([1, 2, 3, 4]).transform(lambda t: t.map(lambda x: x * 2).filter(lambda x: x > 4)).to_list()
    assert result == [6, 8]

  def test_chained_transformations(self):
    """Test chaining multiple transformations."""
    result = (
      Pipeline([1, 2, 3, 4])
      .transform(lambda t: t.map(lambda x: x * 2))
      .transform(lambda t: t.filter(lambda x: x > 4))
      .to_list()
    )
    assert result == [6, 8]


class TestPipelineTerminalOperations:
  """Test terminal operations that consume the pipeline."""

  def test_each_applies_side_effects(self):
    """Test each applies function to all elements."""
    results = []
    Pipeline([1, 2, 3]).each(lambda x: results.append(x * 2))
    assert results == [2, 4, 6]

  def test_first_gets_n_elements(self):
    """Test first gets specified number of elements."""
    result = Pipeline([1, 2, 3, 4, 5]).first(3)
    assert result == [1, 2, 3]

  def test_first_default_one_element(self):
    """Test first with no argument gets one element."""
    result = Pipeline([1, 2, 3]).first()
    assert result == [1]

  def test_first_with_insufficient_data(self):
    """Test first when requesting more elements than available."""
    result = Pipeline([1, 2]).first(5)
    assert result == [1, 2]

  def test_consume_processes_without_return(self):
    """Test consume processes all elements without returning anything."""
    side_effects = []
    transformer = createTransformer(int).tap(lambda x: side_effects.append(x))
    result = Pipeline([1, 2, 3]).apply(transformer).consume()

    assert result is None
    assert side_effects == [1, 2, 3]


class TestPipelineDataTypes:
  """Test pipeline with various data types."""

  def test_string_processing(self):
    """Test pipeline with string data."""
    result = Pipeline(["hello", "world"]).transform(lambda t: t.map(lambda x: x.upper())).to_list()
    assert result == ["HELLO", "WORLD"]

  def test_mixed_types_processing(self):
    """Test pipeline with mixed data types."""
    result = Pipeline([1, "hello", 3.14]).transform(lambda t: t.map(lambda x: str(x))).to_list()
    assert result == ["1", "hello", "3.14"]

  def test_complex_objects_processing(self):
    """Test pipeline with complex objects."""
    data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    result = Pipeline(data).transform(lambda t: t.map(lambda x: x["name"])).to_list()
    assert result == ["Alice", "Bob"]


class TestPipelineEdgeCases:
  """Test edge cases and boundary conditions."""

  def test_empty_pipeline(self):
    """Test pipeline with empty data."""
    assert Pipeline([]).to_list() == []

    # Test terminal operations on empty pipeline
    results = []
    Pipeline([]).each(lambda x: results.append(x))
    assert results == []

    assert Pipeline([]).first(5) == []
    assert Pipeline([]).consume() is None

  def test_single_element_pipeline(self):
    """Test pipeline with single element."""
    assert Pipeline([42]).to_list() == [42]

  def test_type_preservation(self):
    """Test that pipeline preserves and transforms types correctly."""
    # Integers preserved
    int_result = Pipeline([1, 2, 3]).to_list()
    assert all(isinstance(x, int) for x in int_result)

    # Transform to strings
    str_result = Pipeline([1, 2, 3]).transform(lambda t: t.map(lambda x: str(x))).to_list()
    assert all(isinstance(x, str) for x in str_result)
    assert str_result == ["1", "2", "3"]


class TestPipelinePerformance:
  """Test pipeline performance and chunking characteristics."""

  def test_large_dataset_processing(self):
    """Test pipeline handles large datasets efficiently."""
    large_data = list(range(10000))
    result = Pipeline(large_data).transform(lambda t: t.map(lambda x: x * 2).filter(lambda x: x % 100 == 0)).to_list()

    # Every 50th element doubled (0, 100, 200, ..., 19800)
    expected = [x * 2 for x in range(0, 10000, 50)]
    assert result == expected

  def test_chunked_processing_consistency(self):
    """Test that chunked processing produces consistent results."""
    # Use small chunk size to test chunking behavior
    transformer = createTransformer(int, chunk_size=10).map(lambda x: x + 1)
    result = Pipeline(list(range(100))).apply(transformer).to_list()

    expected = list(range(1, 101))  # [1, 2, 3, ..., 100]
    assert result == expected

  def test_buffer_with_two_maps(self):
    """Test that buffer function works correctly with two sequential map operations."""
    # Create a pipeline with two map operations and buffering
    data = list(range(10))

    # Track execution order to verify buffering behavior
    execution_order = []

    def first_map(x):
      execution_order.append(f"first_map({x})")
      return x * 2

    def second_map(x):
      execution_order.append(f"second_map({x})")
      return x + 1

    # Apply buffering with 2 workers between two map operations
    result = (
      Pipeline(data)
      .transform(lambda t: t.map(first_map))
      .buffer(2)  # Buffer with 2 workers
      .transform(lambda t: t.map(second_map))
      .to_list()
    )

    # Verify the final result is correct
    expected = [(x * 2) + 1 for x in range(10)]  # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert result == expected

    # Verify both map operations were called for each element
    assert len([call for call in execution_order if "first_map" in call]) == 10
    assert len([call for call in execution_order if "second_map" in call]) == 10

    # Verify all expected values were processed
    first_map_values = [int(call.split("(")[1].split(")")[0]) for call in execution_order if "first_map" in call]
    second_map_values = [int(call.split("(")[1].split(")")[0]) for call in execution_order if "second_map" in call]

    assert sorted(first_map_values) == list(range(10))
    assert sorted(second_map_values) == [x * 2 for x in range(10)]


class TestPipelineBranch:
  """Test pipeline branch method functionality."""

  def test_branch_basic_functionality(self):
    """Test basic branch operation with simple transformers."""
    # Create a pipeline with basic data
    pipeline = Pipeline([1, 2, 3, 4, 5])

    # Create two different branch transformers
    double_branch = createTransformer(int).map(lambda x: x * 2)
    square_branch = createTransformer(int).map(lambda x: x**2)

    # Execute branching
    result = pipeline.branch({"doubled": double_branch, "squared": square_branch})

    # Verify results contain processed items for each branch
    assert "doubled" in result
    assert "squared" in result
    assert len(result) == 2

    # Each branch gets all items from the pipeline:
    # doubled gets all items: [1, 2, 3, 4, 5] -> [2, 4, 6, 8, 10]
    # squared gets all items: [1, 2, 3, 4, 5] -> [1, 4, 9, 16, 25]
    assert sorted(result["doubled"]) == [2, 4, 6, 8, 10]
    assert sorted(result["squared"]) == [1, 4, 9, 16, 25]

  def test_branch_with_empty_input(self):
    """Test branch with empty input data."""
    pipeline = Pipeline([])

    double_branch = createTransformer(int).map(lambda x: x * 2)
    square_branch = createTransformer(int).map(lambda x: x**2)

    result = pipeline.branch({"doubled": double_branch, "squared": square_branch})

    # Should return empty lists for all branches
    assert result == {"doubled": [], "squared": []}

  def test_branch_with_empty_branches_dict(self):
    """Test branch with empty branches dictionary."""
    pipeline = Pipeline([1, 2, 3])

    result = pipeline.branch({})

    # Should return empty dictionary
    assert result == {}

  def test_branch_with_single_branch(self):
    """Test branch with only one branch."""
    pipeline = Pipeline([1, 2, 3, 4])

    triple_branch = createTransformer(int).map(lambda x: x * 3)

    result = pipeline.branch({"tripled": triple_branch})

    assert len(result) == 1
    assert "tripled" in result
    # With only one branch, it gets all items
    assert sorted(result["tripled"]) == [3, 6, 9, 12]

  def test_branch_with_custom_queue_size(self):
    """Test branch with custom queue size parameter."""
    pipeline = Pipeline([1, 2, 3, 4, 5])

    double_branch = createTransformer(int).map(lambda x: x * 2)
    triple_branch = createTransformer(int).map(lambda x: x * 3)

    # Test with a small queue size
    result = pipeline.branch(
      {
        "doubled": double_branch,
        "tripled": triple_branch,
      },
      max_batch_buffer=2,
    )

    # Each branch gets all items regardless of queue size:
    # doubled gets all items: [1, 2, 3, 4, 5] -> [2, 4, 6, 8, 10]
    # tripled gets all items: [1, 2, 3, 4, 5] -> [3, 6, 9, 12, 15]
    assert sorted(result["doubled"]) == [2, 4, 6, 8, 10]
    assert sorted(result["tripled"]) == [3, 6, 9, 12, 15]

  def test_branch_with_three_branches(self):
    """Test branch with three branches to verify fan-out distribution."""
    pipeline = Pipeline([1, 2, 3, 4, 5, 6, 7, 8, 9])

    add_10 = createTransformer(int).map(lambda x: x + 10)
    add_20 = createTransformer(int).map(lambda x: x + 20)
    add_30 = createTransformer(int).map(lambda x: x + 30)

    result = pipeline.branch({"add_10": add_10, "add_20": add_20, "add_30": add_30})

    # Each branch gets all items:
    # add_10 gets all items: [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [11, 12, 13, 14, 15, 16, 17, 18, 19]
    # add_20 gets all items: [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [21, 22, 23, 24, 25, 26, 27, 28, 29]
    # add_30 gets all items: [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [31, 32, 33, 34, 35, 36, 37, 38, 39]
    assert sorted(result["add_10"]) == [11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert sorted(result["add_20"]) == [21, 22, 23, 24, 25, 26, 27, 28, 29]
    assert sorted(result["add_30"]) == [31, 32, 33, 34, 35, 36, 37, 38, 39]

  def test_branch_with_filtering_transformers(self):
    """Test branch with transformers that filter data."""
    pipeline = Pipeline([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Create transformers that filter data
    even_branch = createTransformer(int).filter(lambda x: x % 2 == 0)
    odd_branch = createTransformer(int).filter(lambda x: x % 2 == 1)

    result = pipeline.branch({"evens": even_branch, "odds": odd_branch})

    # Each branch gets all items and then filters:
    # evens gets all items [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> filters to [2, 4, 6, 8, 10]
    # odds gets all items [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> filters to [1, 3, 5, 7, 9]
    assert result["evens"] == [2, 4, 6, 8, 10]
    assert result["odds"] == [1, 3, 5, 7, 9]

  def test_branch_with_multiple_transformations(self):
    """Test branch with complex multi-step transformers."""
    pipeline = Pipeline([1, 2, 3, 4, 5, 6])

    # Complex transformer: filter evens, then double, then add 1
    complex_branch = createTransformer(int).filter(lambda x: x % 2 == 0).map(lambda x: x * 2).map(lambda x: x + 1)

    # Simple transformer: just multiply by 10
    simple_branch = createTransformer(int).map(lambda x: x * 10)

    result = pipeline.branch({"complex": complex_branch, "simple": simple_branch})

    # Each branch gets all items:
    # complex gets all items [1, 2, 3, 4, 5, 6] -> filters to [2, 4, 6] -> [4, 8, 12] -> [5, 9, 13]
    # simple gets all items [1, 2, 3, 4, 5, 6] -> [10, 20, 30, 40, 50, 60]
    assert result["complex"] == [5, 9, 13]
    assert sorted(result["simple"]) == [10, 20, 30, 40, 50, 60]

  def test_branch_with_chunked_data(self):
    """Test branch behavior with data that gets processed in multiple chunks."""
    # Create a dataset large enough to be processed in multiple chunks
    # with a small chunk size
    data = list(range(1, 21))  # [1, 2, 3, ..., 20]
    pipeline = Pipeline(data)

    # Use small chunk size to ensure multiple chunks
    small_chunk_transformer = createTransformer(int, chunk_size=5).map(lambda x: x * 2)
    identity_transformer = createTransformer(int, chunk_size=5)

    result = pipeline.branch({"doubled": small_chunk_transformer, "identity": identity_transformer})

    # Each branch gets all items:
    # doubled gets all items [1, 2, 3, ..., 20] ->
    # [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    # identity gets all items [1, 2, 3, ..., 20] ->
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert sorted(result["doubled"]) == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    assert sorted(result["identity"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

  def test_branch_with_flatten_operation(self):
    """Test branch with flatten operations."""
    pipeline = Pipeline([[1, 2], [3, 4], [5, 6]])

    flatten_branch = createTransformer(list).flatten()
    count_branch = createTransformer(list).map(lambda x: len(x))

    result = pipeline.branch({"flattened": flatten_branch, "lengths": count_branch})

    # Each branch gets all items:
    # flattened gets all items [[1, 2], [3, 4], [5, 6]] -> flattens to [1, 2, 3, 4, 5, 6]
    # lengths gets all items [[1, 2], [3, 4], [5, 6]] -> [2, 2, 2]
    assert sorted(result["flattened"]) == [1, 2, 3, 4, 5, 6]
    assert result["lengths"] == [2, 2, 2]

  def test_branch_is_terminal_operation(self):
    """Test that branch is a terminal operation that consumes the pipeline."""
    pipeline = Pipeline([1, 2, 3, 4, 5])

    # Create a simple transformer
    double_branch = createTransformer(int).map(lambda x: x * 2)

    # Execute branch
    result = pipeline.branch({"doubled": double_branch})

    # Verify the result - each branch gets all items: [1, 2, 3, 4, 5] -> [2, 4, 6, 8, 10]
    assert sorted(result["doubled"]) == [2, 4, 6, 8, 10]

    # Attempt to use the pipeline again should yield empty results
    # since the iterator has been consumed
    empty_result = pipeline.to_list()
    assert empty_result == []

  def test_branch_with_different_chunk_sizes(self):
    """Test branch with transformers that have different chunk sizes."""
    data = list(range(1, 16))  # [1, 2, 3, ..., 15]
    pipeline = Pipeline(data)

    # Different chunk sizes for different branches
    large_chunk_branch = createTransformer(int, chunk_size=10).map(lambda x: x + 100)
    small_chunk_branch = createTransformer(int, chunk_size=3).map(lambda x: x + 200)

    result = pipeline.branch({"large_chunk": large_chunk_branch, "small_chunk": small_chunk_branch})

    # Each branch gets all items:
    # large_chunk gets all items [1, 2, 3, ..., 15] -> [101, 102, 103, ..., 115]
    # small_chunk gets all items [1, 2, 3, ..., 15] -> [201, 202, 203, ..., 215]

    assert sorted(result["large_chunk"]) == [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    assert sorted(result["small_chunk"]) == [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215]

  def test_branch_preserves_data_order_within_chunks(self):
    """Test that branch preserves data order within the final chunk."""
    pipeline = Pipeline([5, 3, 8, 1, 9, 2])

    # Identity transformer should preserve order
    identity_branch = createTransformer(int)
    reverse_branch = createTransformer(int).map(lambda x: -x)

    result = pipeline.branch({"identity": identity_branch, "negated": reverse_branch})

    # Each branch gets all items:
    # identity gets all items: [5, 3, 8, 1, 9, 2] (preserves order)
    # negated gets all items: [5, 3, 8, 1, 9, 2] -> [-5, -3, -8, -1, -9, -2] (preserves order)
    assert result["identity"] == [5, 3, 8, 1, 9, 2]
    assert result["negated"] == [-5, -3, -8, -1, -9, -2]

  def test_branch_with_error_handling(self):
    """Test branch behavior when transformers encounter errors."""
    pipeline = Pipeline([1, 2, 0, 4, 5])

    # Create a transformer that will fail on zero division
    division_branch = createTransformer(int).map(lambda x: 10 // x)
    safe_branch = createTransformer(int).map(lambda x: x * 2)

    # The division_branch should fail when processing 0
    # The current implementation catches exceptions and returns empty lists for failed branches
    result = pipeline.branch({"division": division_branch, "safe": safe_branch})

    # division gets all items [1, 2, 0, 4, 5] -> fails on 0, returns []
    # safe gets all items [1, 2, 0, 4, 5] -> [2, 4, 0, 8, 10]
    assert result["division"] == []  # Error causes empty result
    assert sorted(result["safe"]) == [0, 2, 4, 8, 10]

  def test_branch_context_isolation(self):
    """Test that different branches don't interfere with each other's context."""
    pipeline = Pipeline([1, 2, 3])

    # Create context-aware transformers that modify context
    def context_modifier_a(chunk: list[int], ctx: PipelineContext) -> list[int]:
      ctx["branch_a_processed"] = len(chunk)
      return [x * 2 for x in chunk]

    def context_modifier_b(chunk: list[int], ctx: PipelineContext) -> list[int]:
      ctx["branch_b_processed"] = len(chunk)
      return [x * 3 for x in chunk]

    branch_a = createTransformer(int)._pipe(context_modifier_a)
    branch_b = createTransformer(int)._pipe(context_modifier_b)

    result = pipeline.branch({"branch_a": branch_a, "branch_b": branch_b})

    # Each branch gets all items:
    # branch_a gets all items: [1, 2, 3] -> [2, 4, 6]
    # branch_b gets all items: [1, 2, 3] -> [3, 6, 9]
    assert sorted(result["branch_a"]) == [2, 4, 6]
    assert result["branch_b"] == [3, 6, 9]

    # Context values should reflect the actual chunk sizes processed
    assert pipeline.ctx.get("branch_a_processed") == 3
    assert pipeline.ctx.get("branch_b_processed") == 3
