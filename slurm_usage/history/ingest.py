"""Keeping the cache in step with slurmdbd, in bounded pieces.

sacct holds its whole answer in memory before printing a line (about 7 KB per
allocation, whatever fields are asked for), so history is fetched newest first in
windows sized from the density of the last one, aiming at TARGET_ROWS each. A
window that would still make sacct too big is stopped and split. Each call costs
slurmdbd and sacct a fixed ~0.5 s, so a few large windows are far cheaper than
many small ones; between background windows the loader rests in proportion to
what the last one cost. Rows go to the cache on a second thread (in order, a
bounded queue), so writing one piece overlaps fetching the next.

Coverage is always one interval ending at the last refresh, and only widens
after a piece has arrived whole and been written.
"""

import queue
import random
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from ..model import Status
from .commands import TooLarge, bulk_sacct, stamp, stream_lines
from .environment import Environment, find_origin
from .errors import HistoryError, Progress, ProgressFn
from .records import (ACTIVE_STATES, ALLOCATION_FIELDS, new_delimiter, normalize, read_rows)
from .store import FLAG_CONFIRMED, FLAG_NONE, FLAG_RUNAWAY, FLAG_STALE
from .timeline import DAY, HOUR

OVERLAP = 2 * HOUR  # between pieces: DST and clock skew
LOOKBACK = DAY  # re-read on refresh: late database writes, requeues of finished jobs
LOOKBACK_EVERY = HOUR  # the full LOOKBACK at most this often; otherwise only OVERLAP
# Between pieces of older history loaded in the background, slurmdbd rests PACE times as
# long as it took to answer the last one (until sacct's first line), at least MIN_PAUSE and
# at most PIECE_PAUSE seconds, counted from that answer: the time sacct then spends
# printing counts as rest. One sacct at a time, always.
PIECE_PAUSE = 1.0
PACE = 0.5
MIN_PAUSE = 0.05
EMPTY_GIVE_UP = 365 * DAY  # with no known origin: this much nothing means accounting began
# sacct needs about 7 KB per row it returns (measured 6.6 KB with -X, independent of the
# fields asked for), and a quiet month can be followed by a day with tens of thousands of
# array tasks. Pieces aim at TARGET_ROWS (~200 MB of sacct) from the densest recent
# piece (each older one counting FORGET times less), grow at most GROWTH times from one to
# the next, and a sacct that grows past CHILD_MEMORY anyway is stopped; its window then
# held at least CHILD_MEMORY / ROW_BYTES rows, which sizes the retry. (Simulated over ten
# months of a busy cluster: ~30 calls, 3 of them stopped, peak 512 MB.)
FIRST_PIECE, SMALLEST_PIECE, LARGEST_PIECE = DAY, HOUR // 4, 64 * DAY
TARGET_ROWS = 30000
GROWTH = 4
FORGET = 0.5
CHILD_MEMORY = 512 * 1024 * 1024
ROW_BYTES = 6600
BATCH = 20000  # records per cache transaction: few commits, bounded memory here
WRITE_DEPTH = 4  # pieces (and what follows each) queued for the writer
CHUNK, CHUNK_DEPTH = 5000, 16  # sacct lines handed over at a time; at most ~40 MB waiting
TIMEOUT = 180.0
IDS_PER_QUERY = 400
NO_ORIGIN_YEARS = 10
BEGINNING = 2 * DAY
STALE_GRACE = DAY  # past its time limit, and nothing confirms that it runs
CLOCK_SLACK = HOUR
LIVE_GRACE = 120  # started this shortly before squeue was asked


class _Piece:
    """One sacct answer on its way from the reader to the writer thread, as lines."""

    def __init__(self, separator: str) -> None:
        self.separator = separator
        self.chunks = queue.Queue(CHUNK_DEPTH)  # type: queue.Queue
        self.count = self.bad = 0
        self.ended = False  # the reader's end mark was taken

    def lines(self) -> Iterator[str]:
        while not self.ended:
            chunk = self.chunks.get()
            if chunk is None:
                self.ended = True
                return
            for line in chunk:
                yield line

    def discard(self) -> None:
        for _ in self.lines():
            pass


class _Writer:
    """Parsing and cache writes on a second thread, strictly in order, so that they
    overlap reading the next piece from sacct (and the rest between pieces). Queues are
    bounded, so memory is. After a failed task the rest are skipped (coverage never
    widens over rows that were not written) and the error is raised in the caller."""

    def __init__(self) -> None:
        self._tasks = queue.Queue(WRITE_DEPTH)  # type: queue.Queue
        self._thread = None  # type: Optional[threading.Thread]
        self._error = None  # type: Optional[BaseException]
        self._failed = False  # skip everything until drained

    def submit(self, task: Callable[[], None], piece: Optional[_Piece] = None) -> None:
        """Run ``task`` after everything before it. A task that reads ``piece`` has the
        piece's lines discarded if it is skipped or fails, so the reader never waits."""
        self._raise()
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="history-writer",
                                            daemon=True)
            self._thread.start()
        self._tasks.put((task, piece))

    def drain(self) -> None:
        """Wait until everything submitted is written; raise what failed."""
        if self._thread is not None:
            self._tasks.put(None)
            self._thread.join()
            self._thread = None
        self._failed = False
        self._raise()

    def _run(self) -> None:
        while True:
            item = self._tasks.get()
            if item is None:
                return
            task, piece = item
            if not self._failed:
                try:
                    task()
                except BaseException as error:  # handed to the caller
                    self._error, self._failed = error, True
            if piece is not None and self._failed:
                piece.discard()

    def _raise(self) -> None:
        error, self._error = self._error, None
        if error is not None:
            raise error

    def hand(self, piece: _Piece, chunk: Optional[List[str]]) -> None:
        """Pass lines on (None: the end of the piece)."""
        piece.chunks.put(chunk)


class Ingestor:
    def __init__(self, runner: Any, store: Any, env: Environment) -> None:
        self.runner, self.store, self.env = runner, store, env
        self.fields = env.supported(ALLOCATION_FIELDS)
        self.malformed = 0
        self._piece = FIRST_PIECE
        self._guarded = True  # whether sacct's memory is watched
        self._density = 0.0  # rows per second expected of the next piece
        self._attempt = (0, 0)  # the window asked for last
        self._heavy = None  # type: Optional[List[float]]  # [start, rows] still owed by a stopped window
        self._last_wait = 0.0  # seconds slurmdbd took to answer the last piece
        self._answered = 0.0  # when it did (time.monotonic)
        self._writer = _Writer()
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
            try:
                self._writer.drain()
            finally:
                if self.coverage():
                    self.flag_open_rows()

    def _tell(self, phase: str, done: int, total: Optional[int], detail: str = "") -> None:
        if self._progress:
            self._progress(Progress(phase, done, total, detail))

    def _fetch(self, options: List[str], wait: bool = False) -> Tuple[int, int]:
        """Read one sacct answer and hand it to the writer thread to parse and store.

        Returns (lines, 0) at once: about the number of records, enough to size the next
        piece. wait=True waits for the writer and returns (records, unreadable)."""
        separator = new_delimiter()
        argv = bulk_sacct(self.fields, separator, "-X", *options)
        began = time.monotonic()
        lines = stream_lines(self.runner, argv, self._cancelled, TIMEOUT,
                             CHILD_MEMORY if self._guarded else None)
        piece = _Piece(separator)
        generation = self._generation
        self._writer.submit(lambda: self._store_piece(piece, generation), piece)
        count, chunk = 0, []  # type: int, List[str]
        answered = None  # type: Optional[float]
        try:
            for line in lines:
                if answered is None:
                    answered = time.monotonic()
                chunk.append(line)
                if len(chunk) >= CHUNK:
                    self._writer.hand(piece, chunk)
                    count, chunk = count + len(chunk), []
            count += len(chunk)
            if chunk:
                self._writer.hand(piece, chunk)
        finally:
            lines.close()
            self._writer.hand(piece, None)  # the rows that did arrive are still good rows
            self._answered = answered or time.monotonic()
            self._last_wait = self._answered - began
        if not wait:
            return count, 0
        self._writer.drain()
        return piece.count, piece.bad

    def _store_piece(self, piece: _Piece, generation: int) -> None:
        """On the writer thread: parse a piece and put its records in the cache."""
        batch, count, bad = [], 0, 0  # type: List[Any], int, int
        for values in read_rows(piece.lines(), piece.separator, self.fields):
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
                self.store.put(batch, generation)
                batch = []
        if batch:
            self.store.put(batch, generation)
        piece.count, piece.bad = count, bad
        self.malformed += bad
        if bad and not count:
            raise HistoryError(Status.ERROR, "sacct output could not be read")

    def _fetch_window(self, start: int, end: int) -> int:
        self._attempt = (start, end)
        until = "now" if end + OVERLAP >= self._now else stamp(end + OVERLAP)
        return self._fetch(["-S", stamp(max(BEGINNING, start - OVERLAP)), "-E", until])[0]

    def _aim(self) -> int:
        """The piece that should hold TARGET_ROWS at the expected density. Rows also count
        the overlaps at both ends and whatever ran across the window."""
        if self._density <= 0:
            return LARGEST_PIECE
        return int(TARGET_ROWS / self._density) - 2 * OVERLAP

    def _resize(self, rows: int, seconds: int) -> None:
        """Aim the next piece at TARGET_ROWS, growing at most GROWTH x."""
        self._guarded = True
        if seconds <= 0:
            return
        self._density = max(rows / float(seconds + 2 * OVERLAP), FORGET * self._density)
        self._piece = max(SMALLEST_PIECE, min(LARGEST_PIECE, GROWTH * self._piece, self._aim()))

    def _shrink(self, error: HistoryError) -> None:
        """After a piece failed: retry a smaller one, if the failure was its size."""
        if isinstance(error, TooLarge):
            if self._piece <= SMALLEST_PIECE:
                self._guarded = False  # so dense that nothing smaller exists: let it through
                return
            # the stopped window held at least this many rows
            least = CHILD_MEMORY / float(ROW_BYTES)
            self._heavy = [self._attempt[0], least]
            floor = least / (self._piece + 2 * OVERLAP)
            self._density = max(self._density, floor)
            self._piece = max(SMALLEST_PIECE, min(self._piece // 2, self._aim()))
            return
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
            if self._now - until < self._piece // 4:  # no sliver left for a call of its own
                until = self._now
            try:
                rows = self._fetch_window(at, until)
            except HistoryError as error:
                self._shrink(error)
                continue
            self._writer.submit(self._settle(at, until) if until > last else self.flag_open_rows)
            self._resize(rows, until - at)
            at = until
            done += rows
            self._tell("refresh", at - begin, self._now - begin, "%d records" % done)

    def _pause(self) -> None:
        """A cancellable rest between background pieces, in proportion to the last one's
        cost, so that slurmdbd is never kept busy by this loader alone. The cache writer
        goes on meanwhile."""
        left = min(PIECE_PAUSE, max(MIN_PAUSE, PACE * self._last_wait))
        if self._answered:
            left -= time.monotonic() - self._answered
        while left > 0 and not self._cancelled():
            time.sleep(min(0.1, left))
            left -= 0.1

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
            if since - target < self._piece // 4:  # no sliver left for a call of its own
                since = target
            try:
                rows = self._fetch_window(since, at)
            except HistoryError as error:
                self._shrink(error)
                continue
            self._writer.submit(self._settle(since, at))
            self._resize(rows, at - since)
            self._owed(rows, since)
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
            self._writer.submit(self._settle(0, at))

    def _owed(self, rows: int, since: int) -> None:
        """Going back through a window that was stopped for its size: what it held and has
        not been read yet is still ahead, so the next piece is sized by that, too."""
        if self._heavy is None:
            return
        self._heavy[1] -= rows
        start, left = self._heavy
        if left <= 0 or since <= start:
            self._heavy = None
            return
        self._piece = max(SMALLEST_PIECE, min(self._piece, int(TARGET_ROWS * (
            since - start + 2 * OVERLAP) / left) - 2 * OVERLAP))

    def _settle(self, start: int, end: int) -> Callable[[], None]:
        """After a piece's rows are written: count it as covered, and flag its open rows
        (re-read rows come back unflagged)."""
        def task() -> None:
            self._widen(start, end)
            self.flag_open_rows()
        return task

    def _refresh_open(self) -> None:
        self._writer.drain()
        stale = [row for row in self.store.open_rows() if row[7] != self._generation]
        jobs = sorted({row[1] for row in stale})
        for at in range(0, len(jobs), IDS_PER_QUERY):
            part = jobs[at:at + IDS_PER_QUERY]
            _, bad = self._fetch(["-j", ",".join(str(job) for job in part)], wait=True)
            if not bad:
                asked = set(part)
                # A clean answer that no longer has the row: the row is gone.
                gone = [r[0] for r in stale if r[1] in asked]
                generation = self._generation
                self._writer.submit(lambda: self.store.delete_unseen(gone, generation))
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
        rows = store.open_rows()
        if not rows:
            return
        dims = store.dims()
        repeated = store.repeated_among(row[1] for row in rows)
        changes = {}  # type: Dict[str, int]
        for key, job, dim, submit, start, suspended, limit, _, old in rows:
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
