"""Per-execution telemetry; the proxy preserves existing call sites."""
import time
import uuid
from contextvars import ContextVar

class Telemetry:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.metrics = {"total_time": 0, "retrieval_time": 0, "generation_time": 0,
                        "query_rewrite_used": False, "no_relevant_context": False, "error_type": None}
        self.logs = {"sources": []}

    @staticmethod
    def start_timer():
        return time.perf_counter()

    @staticmethod
    def stop_timer(start_time):
        return round(time.perf_counter() - start_time, 4)

_current = ContextVar("lexrag_telemetry", default=None)

class TelemetryProxy:
    def reset(self):
        _current.set(Telemetry())

    @property
    def current(self):
        instance = _current.get()
        if instance is None:
            instance = Telemetry()
            _current.set(instance)
        return instance

    def __getattr__(self, name):
        return getattr(self.current, name)

telemetry = TelemetryProxy()
