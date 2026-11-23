"""Background work for the UI, so drawing never waits on Slurm.

``Loader.get(key, fn)`` returns what is known about ``key`` right now and, if
needed, starts ``fn`` on a worker. A key's value stays visible while it is
being recomputed (stale-while-revalidate). ``Loader(workers=0)`` runs inline,
which is what tests and ``--once`` use.
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Hashable, Optional

from ..model import Status

PENDING, OK, FAILED = "pending", "ok", "failed"


class Entry:
    __slots__ = ("state", "value", "status", "message", "at", "future", "generation", "key")

    def __init__(self) -> None:
        self.state = PENDING
        self.value = None  # type: Any
        self.status = Status.UNKNOWN  # of the failure, a model.Status
        self.message = ""
        self.at = None  # type: Optional[float]
        self.future = None  # type: Optional[Future]
        self.generation = None  # type: Any
        self.key = None  # type: Hashable

    @property
    def ready(self) -> bool:
        return self.at is not None and self.state != FAILED

    @property
    def busy(self) -> bool:
        return self.future is not None


class Loader:
    """``seed``: an object with load(key) and save(key, value) (data.ResultCache): a key
    seen for the first time starts with the value it had last time (marked as needing a
    refresh), and every new value is handed to it."""

    def __init__(self, workers: int = 3, on_done: Optional[Callable[[], None]] = None,
                 seed: Any = None) -> None:
        self.seed = seed
        self._pool = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
        self._entries = {}  # type: Dict[Hashable, Entry]
        self._lock = threading.Lock()
        self.on_done = on_done
        self.generation = 0  # bump to revalidate everything
        self.finished = 0  # answers landed so far: a change means something to draw

    def invalidate(self) -> None:
        self.generation += 1

    def clear(self) -> None:
        """Drop every entry. Work still running finishes into entries nobody reads."""
        with self._lock:
            self._entries = {}
        self.generation += 1

    def forget(self, key: Hashable) -> None:
        with self._lock:
            self._entries.pop(key, None)

    @property
    def busy(self) -> int:
        with self._lock:
            return sum(entry.busy for entry in self._entries.values())

    def peek(self, key: Hashable) -> Optional[Entry]:
        return self._entries.get(key)

    def get(self, key: Hashable, fn: Callable[[], Any], max_age: Optional[float] = None,
            generation: Any = None) -> Entry:
        """``generation``: the caller's own version of the inputs; a change revalidates."""
        wanted = (self.generation, generation)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = Entry()
                entry.key = key
                if self.seed is not None:
                    found = self.seed.load(key)
                    if found is not None:  # last known: shown while the new one computes
                        entry.value, entry.state, entry.at = found, OK, time.time()
                        entry.status = Status.OK
            stale = entry.generation != wanted or (
                max_age is not None and entry.at is not None and time.time() - entry.at > max_age)
            if entry.future is not None or not (stale or entry.at is None):
                return entry
            entry.generation = wanted
            if self._pool is None:
                entry.future = Future()
            else:
                entry.future = self._pool.submit(self._run, entry, fn)
                return entry
        self._run(entry, fn)
        return entry

    def _run(self, entry: Entry, fn: Callable[[], Any]) -> None:
        try:
            value, failure = fn(), None
        except Exception as error:  # every failure becomes a placeholder, never a crash
            value, failure = None, error
        with self._lock:  # get() assigns entry.future under this lock
            if failure is None:
                entry.value, entry.state = value, OK
                entry.status, entry.message = Status.OK, ""
            else:
                entry.status = getattr(failure, "status", Status.ERROR)
                entry.message = (str(getattr(failure, "message", None) or failure)
                                 or type(failure).__name__)
                entry.state = FAILED
            entry.at = time.time()
            entry.future = None
            self.finished += 1
        if failure is None and self.seed is not None:
            try:
                self.seed.save(entry.key, value)
            except Exception:
                pass
        if self.on_done is not None:
            try:
                self.on_done()
            except Exception:
                pass

    def drop_queued(self) -> int:
        """Forget work that has not started (the screen it was for is gone): what is on
        screen next is asked for again and goes first. Returns how many were dropped."""
        dropped = 0
        with self._lock:
            for entry in self._entries.values():
                future = entry.future
                if future is not None and future.cancel():
                    entry.future = None
                    entry.generation = None  # asked again when next on screen
                    dropped += 1
        return dropped

    def submit(self, fn: Callable[[], Any]) -> None:
        """Fire and forget."""
        if self._pool is None:
            try:
                fn()
            except Exception:
                pass
        else:
            self._pool.submit(fn)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
