# Assuming the classes from your latest example are in a file named `pipeline_lib.py`
# This includes Pipeline, Transformer, and your HTTPTransformer.
import requests_mock

from laygo import Pipeline
from laygo import create_http_transformer
from laygo.context.simple import SimpleContextManager


class TestHTTPTransformer:
  """
  Test suite for the HTTPTransformer class.
  """

  def test_distributed_transformer_with_mock(self):
    """
    Tests the HTTPTransformer by mocking the worker endpoint.
    This test validates that the client-side of the transformer correctly
    calls the endpoint and processes the response from the (mocked) worker.
    """
    # 1. Define the transformer's properties
    base_url = "http://mock-worker.com"
    endpoint = "/process/data"
    worker_url = f"{base_url}{endpoint}"

    # 2. Define the transformer and its logic using the chainable API.
    # This single instance holds both the client and server logic.
    http_transformer, get_route = create_http_transformer(int, endpoint=base_url)

    http_transformer.map(lambda x: x * 2).filter(lambda x: x > 10)

    # Set a small chunk_size to ensure the client makes multiple requests
    http_transformer.chunk_size = 4

    # 3. Get the worker's logic from the transformer itself
    # The `get_route` method provides the exact function the worker would run.
    _, worker_view_func = get_route()

    # 4. Configure the mock endpoint to use the real worker logic
    def mock_response(request, context):
      """The behavior of the mocked Flask endpoint."""
      input_chunk = request.json()
      # Call the actual view function logic obtained from get_route()
      # We pass None for the context as it's not used in this simple case.
      output_chunk = worker_view_func(input_chunk, SimpleContextManager())
      return output_chunk

    # Use requests_mock context manager
    with requests_mock.Mocker() as m:
      m.post(worker_url, json=mock_response)

      # 5. Run the standard Pipeline with the configured transformer
      initial_data = list(range(10))  # [0, 1, 2, ..., 9]
      pipeline = Pipeline(initial_data).apply(http_transformer)
      result, _ = pipeline.to_list()

      # 6. Assert the final result
      expected_result = [12, 14, 16, 18]
      assert sorted(result) == sorted(expected_result)
