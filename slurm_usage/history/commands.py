"""Running accounting queries through the read-only runner."""

import threading
import time
from typing import Any, Callable, Iterator, List, Sequence

from ..model import Status
from .errors import Cancelled, HistoryError
from .records import NEVER_BULK

_CANCEL_POLL = 0.2


def stamp(when: int) -> str:
    """sacct reads times in local time, so one hour a year is ambiguous: callers overlap."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(when))


def bulk_sacct(fields: Sequence[str], separator: str, *options: str) -> List[str]:
    forbidden = NEVER_BULK.intersection(fields)
    if forbidden:
        raise ValueError("never read in bulk: %s" % ", ".join(sorted(forbidden)))
    return (["sacct", "-a", "-D", "-n", "-P", "--delimiter=" + separator] + list(options)
            + ["-o", ",".join(fields)])


def stream_lines(runner: Any, argv: List[str], cancelled: Callable[[], bool],
                 timeout: float) -> Iterator[str]:
    """Lines of a successful command. Cancellation also works before the first line.

    Raises HistoryError unless the whole output arrived, so a truncated answer
    is never recorded as complete.
    """
    if cancelled():
        raise Cancelled()
    stream = runner.stream(argv, timeout=timeout)
    finished = threading.Event()

    def watch() -> None:
        while not finished.wait(_CANCEL_POLL):
            if cancelled():
                stream.close()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with stream:
            try:
                for line in stream:
                    yield line
            except (OSError, ValueError):
                if not cancelled():
                    raise
    finally:
        finished.set()
    watcher.join()
    if cancelled():
        raise Cancelled()
    result = stream.result
    if result is None:
        raise HistoryError(Status.ERROR, "%s did not finish" % argv[0])
    if result.status not in (Status.OK, Status.EMPTY):
        raise HistoryError(result.status, result.reason)
