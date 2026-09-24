"""Progress events. Every stage reports itself as one of the 18 numbered pipeline steps.

emit(step, name, status, progress, **fields) receives each event; the API streams them to the
website as server-sent events.
"""


class Step:
    def __init__(self, emit, number, name):
        self.emit, self.number, self.name = emit, number, name

    def running(self, progress, **fields):
        self.emit(self.number, self.name, "running", progress, **fields)

    def done(self, progress, **fields):
        self.emit(self.number, self.name, "done", progress, **fields)


def with_note(note, text):
    """Prefix a retry note ("Retry 1/2: ...") to a step detail."""
    return (f"{note} · " if note else "") + text
