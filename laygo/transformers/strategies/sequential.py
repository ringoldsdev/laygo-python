from laygo.transformers.strategies.types import ExecutionStrategy


class SequentialStrategy[In, Out](ExecutionStrategy[In, Out]):
  def execute(self, transformer_logic, chunk_generator, data, context):
    # Logic from the original Transformer.__call__
    for chunk in chunk_generator(data):
      yield from transformer_logic(chunk, context)
