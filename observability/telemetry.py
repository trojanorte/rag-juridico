import time
import uuid


class Telemetry:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.metrics = {}
        self.logs = {}

    def reset(self):
        self.trace_id = str(uuid.uuid4())
        self.metrics = {
            "total_time": 0,
            "retrieval_time": 0,
            "generation_time": 0,
            "chunks": 0,
            "error": None,
        }
        self.logs = {
            "question": "",
            "answer": "",
            "sources": [],
            "context": "",
            "prompt": "",
        }

    def start_timer(self):
        return time.perf_counter()

    def stop_timer(self, start_time):
        return round(time.perf_counter() - start_time, 4)


telemetry = Telemetry()
telemetry.reset()