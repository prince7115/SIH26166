"""One registration at a time, run in a worker thread; its events are buffered for the SSE stream."""
import json
import threading
import time
import traceback


class RunManager:
    def __init__(self, poll_s=0.2):
        self._lock = threading.Lock()
        self._events = []
        self._running = False
        self._poll_s = poll_s

    def start(self, job):
        """Run job(emit) in a worker thread. False if a run is already in progress."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._events = []
        threading.Thread(target=self._work, args=(job,), daemon=True).start()
        return True

    def emit(self, step, name, status, progress, **fields):
        event = {"step": step, "name": name, "status": status, "progress": progress, **fields}
        with self._lock:
            self._events.append(event)

    def _work(self, job):
        try:
            job(self.emit)
        except Exception as e:
            traceback.print_exc()
            self.emit(0, "Error", "error", 0, detail=f"{type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._running = False

    def stream(self):
        """Server-sent events of the current run, until it finishes and every event is sent."""
        sent = 0
        while True:
            with self._lock:
                pending = self._events[sent:]
                finished = not self._running
            for event in pending:
                yield f"data: {json.dumps(event, default=str)}\n\n"
            sent += len(pending)
            if finished:
                return
            time.sleep(self._poll_s)
