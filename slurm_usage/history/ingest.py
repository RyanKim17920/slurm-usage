"""Keeping the cache in step with slurmdbd, in bounded pieces.

sacct's own memory grows with the number of records it returns, so history is
fetched in windows that shrink and grow with their row count, newest first.
Coverage is always one interval ending at the last refresh, and only widens
after a piece has arrived whole.
"""

import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..model import Status
from .commands import bulk_sacct, stamp, stream_lines
from .environment import Environment, find_origin
from .errors import HistoryError, Progress, ProgressFn
from .records import (ACTIVE_STATES, ALLOCATION_FIELDS, new_delimiter, normalize, read_rows)
from .store import FLAG_CONFIRMED, FLAG_NONE, FLAG_RUNAWAY, FLAG_STALE
from .timeline import DAY, HOUR

OVERLAP = 2 * HOUR  # between pieces: DST and clock skew
LOOKBACK = DAY  # re-read on refresh: late database writes, requeues of finished jobs
LOOKBACK_EVERY = HOUR  # the full LOOKBACK at most this often; otherwise only OVERLAP
PIECE_PAUSE = 1.0  # between pieces of older history loaded in the background
EMPTY_GIVE_UP = 365 * DAY  # with no known origin: this much nothing means accounting began
# sacct holds its whole answer in memory, roughly 7 KB per row, and a quiet month can
# be followed by a day with tens of thousands of array tasks. So pieces stay short.
FIRST_PIECE, SMALLEST_PIECE, LARGEST_PIECE = DAY // 2, HOUR // 4, DAY
TARGET_ROWS = 12000
NEARLY_EMPTY, EMPTY_STRIDE = 200, 64 * DAY
BATCH = 5000
TIMEOUT = 180.0
IDS_PER_QUERY = 400
NO_ORIGIN_YEARS = 10
BEGINNING = 2 * DAY
STALE_GRACE = DAY  # past its time limit, and nothing confirms that it runs
CLOCK_SLACK = HOUR
LIVE_GRACE = 120  # started this shortly before squeue was asked


class Ingestor:
    def __init__(self, runner: Any, store: Any, env: Environment) -> None:
        self.runner, self.store, self.env = runner, store, env
        self.fields = env.supported(ALLOCATION_FIELDS)
        self.malformed = 0
        self._piece = FIRST_PIECE
        self._now = 0
        self._generation = 0
        self._live = None  # type: Optional[Set[int]]
        self._progress = None  # type: ProgressFn
        self._cancelled = lambda: False  # type: Callable[[], bool]

    def coverage(self) -> Optional[Tuple[int, int]]:
        """(from, to). from=0 means all of history; ``to`` is when Slurm was last asked."""
        found = self.store.get_meta("coverage")
        return (found[0], found[1]) if found else None

    def update(self, low: Optional[int], now: int, refresh: bool = True,
               progress: ProgressFn = None,
               cancelled: Callable[[], bool] = lambda: False) -> None:
        """Make coverage reach back to ``low`` (None = everything) and, if asked, up to now."""
        self._progress, self._cancelled = progress, cancelled
        known = self.coverage()
        if known and known[1] > now:
            if known[1] > now + CLOCK_SLACK:  # a cache from the future is disposable
                self.store.clear()
                known = None
            else:  # login nodes that share a cache need not agree to the second
                now = known[1]
        backfill = known is None or (known[0] > 0 if low is None else known[0] > low)
        if not backfill and not refresh:
            return
        self._now, self._generation = now, random.getrandbits(40) + 1
        # Before sacct: a job missing here but running there has then really gone.
        self._live = self._live_jobs()
        try:
            if known and refresh:
                self._forward(known[1])
            if backfill:
                self._backward(low, paced=not refresh)
                if not known:  # the first load just read up to now
                    self.store.set_meta("lookback_at", self._now)
            if refresh:
                self._refresh_open()
        finally:
            if self.coverage():
                self.flag_open_rows()

    def _tell(self, phase: str, done: int, total: Optional[int], detail: str = "") -> None:
        if self._progress:
            self._progress(Progress(phase, done, total, detail))

    def _fetch(self, options: List[str]) -> Tuple[int, int]:
        separator = new_delimiter()
        argv = bulk_sacct(self.fields, separator, "-X", *options)
        lines = stream_lines(self.runner, argv, self._cancelled, TIMEOUT)
        batch, count, bad = [], 0, 0
        try:
            for values in read_rows(lines, separator, self.fields):
                if values is None:
                    bad += 1
                    continue
                cluster = values.get("Cluster")
                if cluster and self.env.cluster and cluster.lower() != self.env.cluster.lower():
                    continue  # a federation sibling
                record = normalize(values, self._now)
                if record is None:
                    bad += "." not in values.get("JobIDRaw", "")
                    continue
                batch.append(record)
                count += 1
                if len(batch) >= BATCH:
                    self.store.put(batch, self._generation)
                    batch = []
            if batch:
                self.store.put(batch, self._generation)
        finally:
            lines.close()
        self.malformed += bad
        if bad and not count:
            raise HistoryError(Status.ERROR, "sacct output could not be read")
        return count, bad

    def _fetch_window(self, start: int, end: int) -> int:
        until = "now" if end + OVERLAP >= self._now else stamp(end + OVERLAP)
        return self._fetch(["-S", stamp(max(BEGINNING, start - OVERLAP)), "-E", until])[0]

    def _resize(self, rows: int, seconds: int) -> None:
        """Aim the next piece at TARGET_ROWS from this one's density, growing at most 2x."""
        if seconds <= 0:
            return
        wanted = TARGET_ROWS * seconds // max(rows, 1)
        # Only a nearly empty stretch (before accounting began, an idle cluster) may
        # stride further than a day.
        largest = EMPTY_STRIDE if rows < NEARLY_EMPTY else LARGEST_PIECE
        self._piece = max(SMALLEST_PIECE, min(largest, 2 * self._piece, wanted))

    def _shrink(self, error: HistoryError) -> None:
        if error.status != Status.TIMEOUT or self._piece <= SMALLEST_PIECE:
            raise error
        self._piece = max(SMALLEST_PIECE, self._piece // 4)

    def _widen(self, start: int, end: int) -> None:
        known = self.coverage()
        if known and start <= known[1] + 1 and end >= known[0]:  # another process may have moved it
            start, end = min(start, known[0]), max(end, known[1])
        elif known and known[1] > end:
            return
        self.store.set_meta("coverage", [start, end])

    def _forward(self, last: int) -> None:
        full = self.store.get_meta("lookback_at")
        if full is None or self._now - full >= LOOKBACK_EVERY:
            begin = max(BEGINNING, last - LOOKBACK)
            self.store.set_meta("lookback_at", self._now)
        else:  # new records only; unfinished ones are re-read by id
            begin = max(BEGINNING, last - OVERLAP)
        at, done = begin, 0
        while at < self._now:
            until = min(self._now, at + self._piece)
            try:
                rows = self._fetch_window(at, until)
            except HistoryError as error:
                self._shrink(error)
                continue
            if until > last:
                self._widen(at, until)
            self.flag_open_rows()
            self._resize(rows, until - at)
            at = until
            done += rows
            self._tell("refresh", at - begin, self._now - begin, "%d records" % done)

    def _pause(self) -> None:
        """A short, cancellable rest between background pieces, to spare slurmdbd."""
        for _ in range(int(PIECE_PAUSE * 10)):
            if self._cancelled():
                return
            time.sleep(0.1)

    def _backward(self, low: Optional[int], paced: bool = False) -> None:
        known = self.coverage()
        origin = self.store.get_meta("origin")
        if origin is None:
            origin = find_origin(self.runner) or 0
            if origin:
                self.store.set_meta("origin", origin)
        earliest = origin - DAY if origin else self._now - NO_ORIGIN_YEARS * 365 * DAY
        earliest = max(BEGINNING, min(earliest, self._now - 1))
        to_beginning = low is None or low < earliest
        target = earliest if low is None or low < earliest else min(low, self._now - 1)
        at = top = known[0] if known else self._now
        done = empty = pieces = 0
        while at > target:
            if paced and pieces:
                self._pause()
            pieces += 1
            since = max(target, at - self._piece)
            try:
                rows = self._fetch_window(since, at)
            except HistoryError as error:
                self._shrink(error)
                continue
            self._widen(since, at)
            self.flag_open_rows()  # re-read rows come back unflagged
            self._resize(rows, at - since)
            empty = empty + (at - since) if not rows else 0
            at = since
            done += rows
            if not origin and to_beginning and empty >= EMPTY_GIVE_UP:
                # no recorded origin: a year with nothing at all is taken as the start
                break
            self._tell("history", top - at, top - target, "%d records" % done)
        if to_beginning and at > 0:
            # Everything before accounting began: normally nothing, so one query.
            self._fetch_window(BEGINNING, max(BEGINNING, at))
            self._widen(0, at)

    def _refresh_open(self) -> None:
        stale = [row for row in self.store.open_rows() if row[7] != self._generation]
        jobs = sorted({row[1] for row in stale})
        for at in range(0, len(jobs), IDS_PER_QUERY):
            part = jobs[at:at + IDS_PER_QUERY]
            _, bad = self._fetch(["-j", ",".join(str(job) for job in part)])
            if not bad:
                asked = set(part)
                # A clean answer that no longer has the row: the row is gone.
                self.store.delete_unseen([r[0] for r in stale if r[1] in asked], self._generation)
            self._tell("refresh", min(len(jobs), at + len(part)), len(jobs), "unfinished records")

    def _live_jobs(self) -> Optional[Set[int]]:
        result = self.runner.run(["squeue", "-a", "-h", "-t", "all", "-o", "%A"], timeout=30)
        if result.status not in (Status.OK, Status.EMPTY):
            return None
        return {int(word) for word in result.stdout.split() if word.isascii() and word.isdigit()}

    def flag_open_rows(self) -> None:
        """Decide which rows without an end are really still running."""
        store, live, me = self.store, self._live, self.env.invoker.uid
        known = self.coverage()
        as_of = known[1] if known else self._now
        dims = store.dims()
        repeated = store.repeated_jobs()
        changes = {}  # type: Dict[str, int]
        for key, job, dim, submit, start, suspended, limit, _, old in store.open_rows():
            roles = dims.get(dim)
            if start is None or roles is None or roles.state not in ACTIVE_STATES:
                continue
            flag = FLAG_NONE
            if job in repeated and self._superseded(key, job, roles.uid, submit, start, dims):
                flag = FLAG_RUNAWAY
            elif live is not None and job in live:
                flag = FLAG_CONFIRMED
            elif (live is not None and start <= self._now - LIVE_GRACE
                  and (self.env.live_shows_everyone or roles.uid == me)):
                flag = FLAG_RUNAWAY
            elif limit and limit > 0 and start + suspended + limit + STALE_GRACE < as_of:
                flag = FLAG_STALE
            if flag != old:
                changes[key] = flag
        store.set_flags(changes)

    def _superseded(self, key: str, job: int, uid: Optional[int], submit: Optional[int],
                    start: int, dims: Dict[int, Any]) -> bool:
        for row in self.store.by_job(str(job)):
            if row[0] == key:
                continue
            other = dims.get(row[2])
            if other is None or other.uid != uid:
                continue
            # A requeue gets a later submit time, so two of one job cannot both be running.
            if (row[4] or 0, row[6] or 0) > (submit or 0, start):
                return True
        return False
