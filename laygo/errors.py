from queue import Queue
import threading

from laygo.context.types import IContextManager
from laygo.pipeline import Pipeline
from laygo.types import BaseTransformer


class ErrorHandler:
  """A dedicated pipeline for processing errors in parallel."""

  def __init__(self, transformer: "BaseTransformer[dict, dict]"):
    self._queue = Queue()
    self._transformer = transformer
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def _run(self):
    """The main loop for the error processing thread."""
    Pipeline(self._stream_from_queue()).apply(self._transformer).consume()

  def _stream_from_queue(self):
    """A generator that yields items from the queue."""
    while True:
      item = self._queue.get()
      if item is None:
        break
      yield item

  def handle(self, chunk: list, error: Exception, context: IContextManager):
    """Puts an error into the processing queue."""
    self._queue.put({"chunk": chunk, "error": str(error), "context": context.to_dict()})

  def shutdown(self):
    """Shuts down the error processing pipeline."""
    self._queue.put(None)
    self._thread.join()
