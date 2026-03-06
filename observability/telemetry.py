import time
import uuid


class Telemetry:
    def __init__(self):
        self.reset()

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
            "context": "",
            "sources": [],
            "prompt": "",
        }

    def start_timer(self):
        return time.time()

    def stop_timer(self, start):
        return round(time.time() - start, 3)


telemetry = Telemetry()