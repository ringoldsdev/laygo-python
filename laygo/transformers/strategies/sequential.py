from laygo.transformers.strategies.types import ExecutionStrategy


class SequentialStrategy[In, Out](ExecutionStrategy[In, Out]):
  def execute(self, transformer_logic, chunks, context):
    # Logic from the original Transformer.__call__
    for chunk in chunks:
      yield transformer_logic(chunk, context)
