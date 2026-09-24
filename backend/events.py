"""Pipeline run state shared between the worker thread and the SSE stream."""
import threading

pipeline_lock = threading.Lock()
pipeline_events = []
pipeline_results = {}
state = {"running": False}

def emit_event(step, name, status, progress, **kwargs):
    """Thread-safe event emission."""
    event = {"step": step, "name": name, "status": status, "progress": progress}
    event.update(kwargs)
    with pipeline_lock:
        pipeline_events.append(event)

def reset():
    with pipeline_lock:
        pipeline_events.clear()
        pipeline_results.clear()
