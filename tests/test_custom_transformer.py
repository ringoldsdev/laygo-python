from collections.abc import Iterable
from collections.abc import Iterator

from laygo.context.types import IContextManager
from laygo.pipeline import Pipeline
from laygo.transformers.types import BaseTransformer

# In should be an int


class MultiplierTransformer(BaseTransformer[int, int]):
  def __call__(self, data: Iterable[int], context: IContextManager | None = None) -> Iterator[int]:
    """
    Takes an iterable of data and yields each item multiplied.
    """

    multiplier = context["multiplier"] if context and "multiplier" in context else 1

    for item in data:
      yield item * multiplier


class TestCustomTransformer:
  def test_multiplier_transformer(self):
    data = [1, 2, 3, 4, 5]
    expected_output = [2, 4, 6, 8, 10]

    result, _ = Pipeline(data).context({"multiplier": 2}).apply(MultiplierTransformer()).to_list()

    assert result == expected_output, f"Expected {expected_output}, but got {result}"
