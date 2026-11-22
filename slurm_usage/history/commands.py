"""Running accounting queries through the read-only runner."""

import threading
import time
from typing import Any, Callable, Iterator, List, Optional, Sequence

from ..model import Status
from .errors import Cancelled, HistoryError
from .records import NEVER_BULK

_CANCEL_POLL = 0.05  # also how often a memory limit is checked


class TooLarge(HistoryError):
    """The command was stopped because it grew past the memory it was allowed."""

    def __init__(self, limit: int) -> None:
        super().__init__(Status.PARTIAL, "stopped at %d MB of memory" % (limit >> 20))


def child_memory(stream: Any) -> Optional[int]:
    """Resident bytes of a running command, where the platform tells (Linux /proc)."""
    pid = getattr(getattr(stream, "_proc", None), "pid", None)
    if not isinstance(pid, int):
        return None
    try:
        with open("/proc/%d/status" % pid, "rb") as status:
            for line in status:
                if line.startswith(b"VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


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
                 timeout: float, memory_limit: Optional[int] = None) -> Iterator[str]:
    """Lines of a successful command. Cancellation also works before the first line.

    Raises HistoryError unless the whole output arrived, so a truncated answer
    is never recorded as complete; TooLarge if the command's resident memory went
    past ``memory_limit`` bytes (checked only where the platform tells).
    """
    if cancelled():
        raise Cancelled()
    stream = runner.stream(argv, timeout=timeout)
    finished = threading.Event()
    too_large = []  # type: List[bool]

    def watch() -> None:
        while not finished.wait(_CANCEL_POLL):
            if cancelled():
                stream.close()
                return
            if memory_limit is not None:
                used = child_memory(stream)
                if used is not None and used > memory_limit:
                    too_large.append(True)
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
                if not cancelled() and not too_large:
                    raise
    finally:
        finished.set()
    watcher.join()
    if cancelled():
        raise Cancelled()
    if too_large:
        raise TooLarge(memory_limit or 0)
    result = stream.result
    if result is None:
        raise HistoryError(Status.ERROR, "%s did not finish" % argv[0])
    if result.status not in (Status.OK, Status.EMPTY):
        raise HistoryError(result.status, result.reason)
