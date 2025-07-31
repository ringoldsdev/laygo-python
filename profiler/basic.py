from laygo.pipeline import Pipeline
from laygo.transformers.transformer import create_transformer


def int_generator():
  yield from range(10000000)


transformer = create_transformer(int, chunk_size=100000).map(lambda x: x * 2).filter(lambda x: x % 10 == 0)

result, _ = Pipeline(int_generator()).apply(transformer).consume()
