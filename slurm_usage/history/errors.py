"""Failures the history layer raises."""

from typing import Any, Callable, NamedTuple, Optional


class HistoryError(Exception):
    """A query that could not be answered; ``status`` is a model.Status value."""

    def __init__(self, status: str, message: str = "") -> None:
        super().__init__(message or status)
        self.status = status
        self.message = message or status


class Cancelled(Exception):
    """The caller asked to stop. Whatever was already committed stays valid."""


class Note(NamedTuple):
    """Something the front end must say next to the numbers."""

    code: str
    message: str


class Progress(NamedTuple):
    phase: str  # "history", "refresh", "efficiency", "aggregate"
    done: int
    total: Optional[int] = None
    detail: str = ""


ProgressFn = Optional[Callable[[Progress], None]]


def canceller(cancel: Any) -> Callable[[], bool]:
    """Accept None, a threading.Event or a callable."""
    if cancel is None:
        return lambda: False
    if hasattr(cancel, "is_set"):
        return cancel.is_set
    return cancel
