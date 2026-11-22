"""The disposable history cache: numeric accounting fields, names and roles.

Submit lines, scripts, paths and comments are never written here. The store is
SQLite in a private directory, SQLite in memory if that cannot be had, or plain
Python objects if this Python has no sqlite3.

Every public method is safe to call from any thread.
"""

import hashlib
import json
import os
import stat
import threading
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

from .errors import HistoryError, Note
from .records import Record, Roles

try:
    import sqlite3
except ImportError:  # some Python builds have no sqlite3
    sqlite3 = None  # type: ignore

SCHEMA = 4  # 4: 64 KiB pages
OPEN_STOP = 2 ** 40  # "stop" of a row with no end; later than any real time

# jobs.flag, only meaningful while a row has no end
FLAG_NONE, FLAG_RUNAWAY, FLAG_STALE, FLAG_CONFIRMED = 0, 1, 2, 3


class Row(NamedTuple):
    """What a scan yields, ordered by start."""

    key: str
    job: int
    dim: int
    nodes: int
    submit: Optional[int]
    eligible: Optional[int]
    start: int
    end: Optional[int]
    suspended: int
    cpus: Optional[int]
    gpus: Optional[int]
    mem: Optional[int]
    limit: Optional[int]
    flag: int
    used_cpu_ms: Optional[int]
    used_rss: Optional[int]
    gpu_util: Optional[float]
    checked: bool  # steps were read, whatever they held


class Info(NamedTuple):
    label: str
    name: str
    restarts: Optional[int]


_TABLES = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dims(id INTEGER PRIMARY KEY, v TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS nodesets(id INTEGER PRIMARY KEY, v TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(
  k TEXT PRIMARY KEY, job INTEGER, label TEXT, name TEXT, dim INTEGER, nodes INTEGER,
  submit INTEGER, eligible INTEGER, start INTEGER, stop INTEGER, susp INTEGER,
  cpus INTEGER, gpus INTEGER, mem INTEGER, lim INTEGER, restarts INTEGER,
  flag INTEGER, gen INTEGER, ex TEXT);
CREATE INDEX IF NOT EXISTS jobs_stop ON jobs(stop);
CREATE INDEX IF NOT EXISTS jobs_start ON jobs(start);
CREATE INDEX IF NOT EXISTS jobs_job ON jobs(job);
CREATE INDEX IF NOT EXISTS jobs_label ON jobs(label);
CREATE TABLE IF NOT EXISTS usage(k TEXT PRIMARY KEY, cpu INTEGER, rss INTEGER, util REAL);
CREATE TABLE IF NOT EXISTS rollup(
  hour INTEGER, dim INTEGER, gpu INTEGER, cpu INTEGER, mem_mib INTEGER, mem_rest INTEGER,
  attempts INTEGER,
  cpu_cov INTEGER, cpu_used REAL, mem_cov INTEGER, mem_used REAL, gpu_cov INTEGER,
  gpu_used REAL, PRIMARY KEY(hour, dim)) WITHOUT ROWID;
"""
_COLUMNS = ("j.k,j.job,j.dim,j.nodes,j.submit,j.eligible,j.start,"
            "NULLIF(j.stop,%d),j.susp,j.cpus,j.gpus,j.mem,j.lim,j.flag" % OPEN_STOP)
_MAX_SQL_IDS = 5000
PAGE_SIZE = 65536
CACHE_KIB = 32768  # SQLite's page cache per connection
_MIB_MASK = (1 << 20) - 1


class _Base:
    persistent = False
    path = None  # type: Optional[str]

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.notes = []  # type: List[Note]
        self.version = 0  # changes whenever the data does
        self._forget()

    def _forget(self) -> None:
        self._dim_ids = {}  # type: Dict[Roles, int]
        self._set_ids = {}  # type: Dict[str, int]
        self._repeated = (-1, set())  # type: Tuple[int, Set[int]]

    def repeated_jobs(self) -> Set[int]:
        """Job ids with more than one row: the only ones a distinct count must remember."""
        with self.lock:
            if self._repeated[0] != self.version:
                self._repeated = (self.version, self._find_repeated())
            return self._repeated[1]

    def _find_repeated(self) -> Set[int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SqliteStore(_Base):
    def __init__(self, path: str = ":memory:") -> None:
        super().__init__()
        self.path = None if path == ":memory:" else path
        self.persistent = self.path is not None
        self._db = sqlite3.connect(path, timeout=15, check_same_thread=False)
        try:
            self._tune()
            self._db.executescript(_TABLES)
            self._db.commit()
            self._db.execute("SELECT count(*) FROM meta").fetchone()
        except sqlite3.Error:
            self._db.close()
            raise

    def _tune(self) -> None:
        """Fewer, larger writes: what costs on a network home directory is the number of
        calls, not the bytes (measured on virtiofs: 64 KiB pages and no fsync made a cold
        write about 8x faster). The cache is disposable, so it does without fsync: a crash
        of this process loses nothing (the journal stays), a crash of the machine may cost
        the cache, which is then rebuilt. Each setting is best effort."""
        pragmas = ["cache_size=-%d" % CACHE_KIB]
        if self.persistent:
            # page_size only takes effect on a new, empty file; TRUNCATE spares the
            # create and delete of a journal file per transaction. Not WAL: caches on
            # network filesystems may be shared by login nodes, where WAL is unsafe.
            pragmas = ["page_size=%d" % PAGE_SIZE] + pragmas + ["journal_mode=TRUNCATE",
                                                               "synchronous=OFF"]
        for pragma in pragmas:
            try:
                self._db.execute("PRAGMA " + pragma).fetchall()
            except sqlite3.OperationalError:  # e.g. another process holds a lock just now
                pass

    def close(self) -> None:
        with self.lock:
            self._db.close()

    def _select(self, sql: str, args: Sequence[Any] = ()) -> List[tuple]:
        with self.lock:
            try:
                return self._db.execute(sql, args).fetchall()
            except sqlite3.Error as error:
                raise HistoryError("error", "history cache: %s" % error)

    def _write(self, work) -> None:
        with self.lock:
            try:
                # One transaction from the first read on: otherwise each SELECT takes and
                # drops its own lock, a round trip apiece on a network filesystem.
                if not self._db.in_transaction:
                    self._db.execute("BEGIN IMMEDIATE")
                work(self._db)
                self._db.commit()
            except sqlite3.Error as error:
                self._db.rollback()
                self._forget()
                raise HistoryError("error", "history cache: %s" % error)
            except BaseException:
                self._db.rollback()
                self._forget()
                raise
            self.version += 1

    def get_meta(self, key: str, default: Any = None) -> Any:
        found = self._select("SELECT v FROM meta WHERE k=?", (key,))
        return json.loads(found[0][0]) if found else default

    def set_meta(self, key: str, value: Any) -> None:
        self._write(lambda db: db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)",
                                          (key, json.dumps(value))))

    def _intern(self, db, table: str, cache: dict, value: Any, text: str) -> int:
        found = cache.get(value)
        if found is None:
            db.execute("INSERT OR IGNORE INTO %s(v) VALUES(?)" % table, (text,))
            found = db.execute("SELECT id FROM %s WHERE v=?" % table, (text,)).fetchone()[0]
            cache[value] = found
        return found

    def put(self, records: Sequence[Record], generation: int) -> None:
        def work(db):
            rows = []
            for r in records:
                dim = self._dim_ids.get(r.roles)
                if dim is None:
                    dim = self._intern(db, "dims", self._dim_ids, r.roles, json.dumps(r.roles))
                nodes = self._set_ids.get(r.nodes)
                if nodes is None:
                    nodes = self._intern(db, "nodesets", self._set_ids, r.nodes, r.nodes)
                rows.append((r.key, r.job, r.label, r.name, dim, nodes, r.submit, r.eligible,
                             r.start, OPEN_STOP if r.end is None else r.end, r.suspended, r.cpus,
                             r.gpus, r.mem, r.limit, r.restarts, FLAG_NONE, generation, r.exit))
            # Re-read finished rows are nearly always unchanged: rewriting them costs a
            # journal write per page, which is slow on network home directories.
            stored = {}  # type: Dict[str, tuple]
            for at in range(0, len(rows), 500):
                part = [row[0] for row in rows[at:at + 500]]
                for found in db.execute("SELECT * FROM jobs WHERE k IN (%s)"
                                        % ",".join("?" * len(part)), part):
                    stored[found[0]] = found
            changed = []
            for row in rows:
                old = stored.get(row[0])
                if (old is not None and row[9] != OPEN_STOP and old[:16] == row[:16]
                        and old[18] == row[18]):
                    continue
                changed.append(row)
            db.executemany("INSERT OR REPLACE INTO jobs VALUES(%s)" % ",".join("?" * 19),
                           changed)
            started = [row for row in changed if row[8] is not None]
            # the span a changed row used to cover changes too
            started += [stored[row[0]] for row in changed
                        if row[0] in stored and stored[row[0]][8] is not None]
            if started:
                self._dirty(db, min(row[8] for row in started), max(row[9] for row in started))

        self._write(work)

    def _dirty(self, db, low: Optional[int], high: Optional[int]) -> None:
        """Widen the span whose rollup must be rebuilt."""
        if low is None or high is None:
            return
        found = db.execute("SELECT v FROM meta WHERE k='dirty'").fetchone()
        if found:
            old = json.loads(found[0])
            low, high = min(low, old[0]), max(high, old[1])
        db.execute("INSERT OR REPLACE INTO meta VALUES('dirty',?)", (json.dumps([low, high]),))

    def _dirty_keys(self, db, keys: Sequence[str]) -> None:
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            low, high = db.execute("SELECT min(start), max(stop) FROM jobs WHERE k IN (%s) "
                                   "AND start IS NOT NULL" % ",".join("?" * len(part)),
                                   part).fetchone()
            self._dirty(db, low, high)

    def rollup_replace(self, low: int, high: int, rows: Iterable[tuple],
                       dirty: Optional[Tuple[int, int]]) -> None:
        """Replace hours [low, high) and record what is still dirty."""
        def work(db):
            db.execute("DELETE FROM rollup WHERE hour>=? AND hour<?", (low, high))
            # byte-seconds outgrow SQLite's 64-bit sums, so memory is kept as MiB + rest
            db.executemany("INSERT INTO rollup VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           [r[:4] + (r[4] >> 20, r[4] & _MIB_MASK) + tuple(r[5:]) for r in rows])
            if dirty is None:
                db.execute("DELETE FROM meta WHERE k='dirty'")
            else:
                db.execute("INSERT OR REPLACE INTO meta VALUES('dirty',?)", (json.dumps(dirty),))

        self._write(work)

    def rollup_sum(self, low: int, high: int) -> List[tuple]:
        """(dim, gpu, cpu, mem, attempts, cpu_cov, cpu_used, mem_cov, mem_used, gpu_cov, gpu_used)."""
        found = self._select(
            "SELECT dim,sum(gpu),sum(cpu),sum(mem_mib),sum(mem_rest),sum(attempts),sum(cpu_cov),"
            "sum(cpu_used),total(mem_cov),total(mem_used),sum(gpu_cov),sum(gpu_used) "
            "FROM rollup WHERE hour>=? AND hour<? GROUP BY dim", (low, high))
        return [r[:3] + (((r[3] or 0) << 20) + (r[4] or 0),) + tuple(r[5:]) for r in found]

    def rollup_series(self, low: int, high: int, step: int) -> List[Tuple[int, int, int]]:
        """(bins back from ``high``, gpu, cpu) for hours in [low, high); bin 0 ends at high."""
        return self._select(
            "SELECT (?-1-hour)/?,sum(gpu),sum(cpu) FROM rollup WHERE hour>=? AND hour<? "
            "GROUP BY 1", (high, step, low, high))

    def rollup_activity(self) -> List[Tuple[int, int, int]]:
        """(dim, first hour, last hour) with any usage."""
        return self._select("SELECT dim,min(hour),max(hour) FROM rollup GROUP BY dim")

    def count_by_dim(self) -> Dict[int, int]:
        return dict(self._select("SELECT dim,count(*) FROM jobs GROUP BY dim"))

    def cheaper_by_start(self, low: int, high: int) -> bool:
        """Whether rows overlapping [low, high) are fewer through the start index."""
        before = self._select("SELECT count(*) FROM jobs WHERE start<?", (high,))[0][0]
        after = self._select("SELECT count(*) FROM jobs WHERE stop>=?", (low,))[0][0]
        return before < after

    def dims(self) -> Dict[int, Roles]:
        return {i: Roles(*json.loads(v)) for i, v in self._select("SELECT id,v FROM dims")}

    def nodesets(self) -> Dict[int, str]:
        return dict(self._select("SELECT id,v FROM nodesets"))

    def count(self) -> int:
        return self._select("SELECT count(*) FROM jobs")[0][0]

    def has_gpus(self) -> bool:
        return bool(self._select("SELECT 1 FROM jobs WHERE gpus>0 LIMIT 1"))

    def first_start(self, dim_ids: Optional[Set[int]] = None) -> Optional[int]:
        if dim_ids is not None and not dim_ids:
            return None
        where = ""
        if dim_ids is not None and len(dim_ids) <= _MAX_SQL_IDS:
            where = " WHERE dim IN (%s)" % ",".join(str(int(i)) for i in dim_ids)
        return self._select("SELECT min(start) FROM jobs" + where)[0][0]

    def scan(self, low: Optional[int], high: int, dim_ids: Optional[Set[int]] = None,
             usage: bool = False, ordered: bool = True, by_start: bool = False,
             extra: bool = False) -> Iterator[Row]:
        """Allocations that started before ``high`` and had not ended before ``low``.

        extra=True appends (name, exit) to each row.
        """
        if dim_ids is not None and not dim_ids:
            return
        where, args = ["j.start<?"], [high]
        if low is not None:
            # "+" chooses the index: stop for recent windows, start for old ones
            if by_start:
                where = ["j.start<?", "+j.stop>=?"]
            else:
                where = ["+j.start<?", "j.stop>=?"]
            args.append(low)
        check = None
        if dim_ids is not None:
            if len(dim_ids) <= _MAX_SQL_IDS:
                where.append("j.dim IN (%s)" % ",".join(str(int(i)) for i in dim_ids))
            else:
                check = dim_ids
        sql = "SELECT %s%s%s FROM jobs j%s WHERE %s%s" % (
            _COLUMNS,
            ",u.cpu,u.rss,u.util,u.k IS NOT NULL" if usage else ",NULL,NULL,NULL,0",
            ",j.name,j.ex" if extra else "",
            " LEFT JOIN usage u ON u.k=j.k" if usage else "",
            " AND ".join(where),
            " ORDER BY +j.start" if ordered else "")
        with self.lock:
            try:
                for row in self._db.execute(sql, args):
                    if check is None or row[2] in check:
                        yield row  # type: ignore  # same layout as Row, cheaper as a tuple
            except sqlite3.Error as error:
                raise HistoryError("error", "history cache: %s" % error)

    def idle_counts(self, low: Optional[int], high: int) -> List[Tuple[int, bool, int]]:
        """(dim, ended, rows) for records that never started."""
        low = 0 if low is None else low
        return [(dim, bool(ended), n) for dim, ended, n in self._select(
            "SELECT dim, stop<?, count(*) FROM jobs WHERE start IS NULL AND stop>=? "
            "AND ((stop=? AND (submit IS NULL OR submit<?)) OR stop<=?) GROUP BY 1,2",
            (OPEN_STOP, low, OPEN_STOP, high, high))]

    def _find_repeated(self) -> Set[int]:
        return {r[0] for r in self._select(
            "SELECT job FROM jobs GROUP BY job HAVING count(*)>1")}

    def repeated_among(self, jobs: Iterable[int]) -> Set[int]:
        """Those of ``jobs`` with more than one row; cheaper than repeated_jobs() for a few."""
        jobs, found = sorted(set(jobs)), set()  # type: List[int], Set[int]
        for i in range(0, len(jobs), 500):
            part = jobs[i:i + 500]
            found.update(r[0] for r in self._select(
                "SELECT job FROM jobs WHERE job IN (%s) GROUP BY job HAVING count(*)>1"
                % ",".join("?" * len(part)), part))
        return found

    def info(self, keys: Sequence[str]) -> Dict[str, Info]:
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            for k, label, name, restarts in self._select(
                    "SELECT k,label,name,restarts FROM jobs WHERE k IN (%s)"
                    % ",".join("?" * len(part)), part):
                out[k] = Info(label, name, restarts)
        return out

    def by_job(self, job_id: str) -> List[Row]:
        column, value = "label", job_id  # type: str, Any
        if job_id.isascii() and job_id.isdigit() and int(job_id) < 2 ** 63:
            column, value = "job", int(job_id)
        return [Row(*r) for r in self._select(
            "SELECT %s,u.cpu,u.rss,u.util,u.k IS NOT NULL FROM jobs j LEFT JOIN usage u ON u.k=j.k "
            "WHERE j.%s=? ORDER BY j.submit, j.start" % (_COLUMNS, column), (value,))]

    def open_rows(self) -> List[tuple]:
        """(key, job, dim, submit, start, suspended, limit, generation, flag) of unfinished rows."""
        return self._select("SELECT k,job,dim,submit,start,susp,lim,gen,flag FROM jobs "
                            "WHERE stop=?", (OPEN_STOP,))

    def delete_unseen(self, keys: Iterable[str], generation: int) -> None:
        keys = list(keys)

        def work(db):
            self._dirty_keys(db, keys)
            db.executemany("DELETE FROM jobs WHERE k=? AND gen<>? AND stop=?",
                           [(k, generation, OPEN_STOP) for k in keys])

        self._write(work)

    def set_flags(self, changes: Dict[str, int]) -> None:
        if changes:
            def work(db):
                self._dirty_keys(db, list(changes))
                db.executemany("UPDATE jobs SET flag=? WHERE k=?",
                               [(f, k) for k, f in changes.items()])

            self._write(work)

    def put_usage(self, rows: Iterable[Tuple[str, Optional[int], Optional[int],
                                             Optional[float]]]) -> None:
        rows = list(rows)

        def work(db):
            self._dirty_keys(db, [row[0] for row in rows])
            db.executemany("INSERT OR REPLACE INTO usage VALUES(?,?,?,?)", rows)

        self._write(work)

    def mark_checked(self, keys: Iterable[str]) -> None:
        self._write(lambda db: db.executemany("INSERT OR IGNORE INTO usage(k) VALUES(?)",
                                              [(k,) for k in keys]))

    def unchecked(self, low: Optional[int], high: int) -> List[Tuple[str, int, int, int]]:
        """(key, dim, start, end) of finished allocations whose steps were never read."""
        return self._select(
            "SELECT k,dim,start,stop FROM jobs j WHERE stop>=? AND stop<? AND +start<? "
            "AND stop>start AND NOT EXISTS(SELECT 1 FROM usage u WHERE u.k=j.k) ORDER BY stop",
            (0 if low is None else low, OPEN_STOP, high))

    def clear(self) -> None:
        def work(db):
            for table in ("jobs", "usage", "dims", "nodesets", "meta", "rollup"):
                db.execute("DELETE FROM %s" % table)

        self._write(work)
        self._forget()


class MemoryStore(_Base):
    """The same interface without sqlite3. Nothing survives the process."""

    def _dirty_span(self, low: Optional[int], high: Optional[int]) -> None:
        if low is None or high is None:
            return
        old = self._meta.get("dirty")
        if old:
            low, high = min(low, old[0]), max(high, old[1])
        self._meta["dirty"] = [low, high]

    def _dirty_keys(self, keys: Iterable[str]) -> None:
        for k in keys:
            row = self._rows.get(k)
            if row and row[6] is not None:
                self._dirty_span(row[6], OPEN_STOP if row[7] is None else row[7])

    def rollup_replace(self, low: int, high: int, rows: Iterable[tuple],
                       dirty: Optional[Tuple[int, int]]) -> None:
        with self.lock:
            for key in [key for key in self._rollup if low <= key[0] < high]:
                del self._rollup[key]
            for row in rows:
                self._rollup[(row[0], row[1])] = tuple(row[2:])
            if dirty is None:
                self._meta.pop("dirty", None)
            else:
                self._meta["dirty"] = list(dirty)
            self.version += 1

    def rollup_sum(self, low: int, high: int) -> List[tuple]:
        sums = {}  # type: Dict[int, list]
        with self.lock:
            for (hour, dim), values in self._rollup.items():
                if low <= hour < high:
                    found = sums.setdefault(dim, [0] * len(values))
                    for at, value in enumerate(values):
                        found[at] += value
        return [(dim,) + tuple(values) for dim, values in sums.items()]

    def rollup_series(self, low: int, high: int, step: int) -> List[Tuple[int, int, int]]:
        sums = {}  # type: Dict[int, List[int]]
        with self.lock:
            for (hour, dim), values in self._rollup.items():
                if low <= hour < high:
                    found = sums.setdefault((high - 1 - hour) // step, [0, 0])
                    found[0] += values[0]
                    found[1] += values[1]
        return [(at, gpu, cpu) for at, (gpu, cpu) in sums.items()]

    def rollup_activity(self) -> List[Tuple[int, int, int]]:
        found = {}  # type: Dict[int, List[int]]
        with self.lock:
            for hour, dim in self._rollup:
                pair = found.setdefault(dim, [hour, hour])
                pair[0], pair[1] = min(pair[0], hour), max(pair[1], hour)
        return [(dim, a, b) for dim, (a, b) in found.items()]

    def count_by_dim(self) -> Dict[int, int]:
        counts = {}  # type: Dict[int, int]
        with self.lock:
            for row in self._rows.values():
                counts[row[2]] = counts.get(row[2], 0) + 1
        return counts

    def cheaper_by_start(self, low: int, high: int) -> bool:
        return False

    def __init__(self) -> None:
        super().__init__()
        self._clear()

    def _clear(self) -> None:
        self._rollup = {}  # type: Dict[Tuple[int, int], tuple]
        self._meta = {}  # type: Dict[str, Any]
        self._rows = {}  # type: Dict[str, tuple]  # Row fields + (label, name, restarts, gen)
        self._usage = {}  # type: Dict[str, tuple]
        self._dims = []  # type: List[Roles]
        self._sets = []  # type: List[str]

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self._meta.get(key, default)

    def set_meta(self, key: str, value: Any) -> None:
        with self.lock:
            self._meta[key] = json.loads(json.dumps(value))
            self.version += 1

    def put(self, records: Sequence[Record], generation: int) -> None:
        with self.lock:
            for r in records:
                if r.roles not in self._dim_ids:
                    self._dim_ids[r.roles] = len(self._dims)
                    self._dims.append(r.roles)
                if r.nodes not in self._set_ids:
                    self._set_ids[r.nodes] = len(self._sets)
                    self._sets.append(r.nodes)
                self._rows[r.key] = (r.key, r.job, self._dim_ids[r.roles], self._set_ids[r.nodes],
                                     r.submit, r.eligible, r.start, r.end, r.suspended, r.cpus,
                                     r.gpus, r.mem, r.limit, FLAG_NONE,
                                     r.label, r.name, r.restarts, generation, r.exit)
                if r.start is not None:
                    self._dirty_span(r.start, OPEN_STOP if r.end is None else r.end)
            self.version += 1

    def dims(self) -> Dict[int, Roles]:
        with self.lock:
            return dict(enumerate(self._dims))

    def nodesets(self) -> Dict[int, str]:
        with self.lock:
            return dict(enumerate(self._sets))

    def count(self) -> int:
        return len(self._rows)

    def has_gpus(self) -> bool:
        with self.lock:
            return any(r[10] for r in self._rows.values())

    def first_start(self, dim_ids: Optional[Set[int]] = None) -> Optional[int]:
        with self.lock:
            return min((r[6] for r in self._rows.values() if r[6] is not None
                        and (dim_ids is None or r[2] in dim_ids)), default=None)

    def scan(self, low: Optional[int], high: int, dim_ids: Optional[Set[int]] = None,
             usage: bool = False, ordered: bool = True, by_start: bool = False,
             extra: bool = False) -> Iterator[Row]:
        empty = (None, None, None, False)
        with self.lock:
            picked = [r for r in self._rows.values()
                      if r[6] is not None and r[6] < high
                      and (low is None or r[7] is None or r[7] >= low)
                      and (dim_ids is None or r[2] in dim_ids)]
            if ordered:
                picked.sort(key=lambda r: r[6])
            for r in picked:
                out = r[:14] + (self._usage.get(r[0], empty) if usage else empty)
                yield out + ((r[15], r[18]) if extra else ())  # type: ignore

    def idle_counts(self, low: Optional[int], high: int) -> List[Tuple[int, bool, int]]:
        counts = {}  # type: Dict[Tuple[int, bool], int]
        with self.lock:
            for r in self._rows.values():
                if r[6] is not None:
                    continue
                if r[7] is None:
                    keep = r[4] is None or r[4] < high
                else:
                    keep = (low or 0) <= r[7] <= high
                if keep:
                    key = (r[2], r[7] is not None)
                    counts[key] = counts.get(key, 0) + 1
        return [(dim, ended, n) for (dim, ended), n in counts.items()]

    def _find_repeated(self) -> Set[int]:
        seen, repeated = set(), set()  # type: Set[int], Set[int]
        for r in self._rows.values():
            (repeated if r[1] in seen else seen).add(r[1])
        return repeated

    def repeated_among(self, jobs: Iterable[int]) -> Set[int]:
        return self.repeated_jobs().intersection(jobs)

    def info(self, keys: Sequence[str]) -> Dict[str, Info]:
        with self.lock:
            return {k: Info(*self._rows[k][14:17]) for k in keys if k in self._rows}

    def by_job(self, job_id: str) -> List[Row]:
        numeric = int(job_id) if job_id.isascii() and job_id.isdigit() else None
        with self.lock:
            picked = [r for r in self._rows.values()
                      if (r[1] == numeric if numeric is not None else r[14] == job_id)]
            picked.sort(key=lambda r: (r[4] or 0, r[6] or 0))
            return [Row(*(r[:14] + self._usage.get(r[0], (None, None, None, False)))) for r in picked]

    def open_rows(self) -> List[tuple]:
        with self.lock:
            return [(r[0], r[1], r[2], r[4], r[6], r[8], r[12], r[17], r[13])
                    for r in self._rows.values() if r[7] is None]

    def delete_unseen(self, keys: Iterable[str], generation: int) -> None:
        with self.lock:
            keys = list(keys)
            self._dirty_keys(keys)
            for k in keys:
                row = self._rows.get(k)
                if row and row[7] is None and row[17] != generation:
                    del self._rows[k]
            self.version += 1

    def _change(self, position: int, changes: Dict[str, int]) -> None:
        with self.lock:
            for k, value in changes.items():
                row = self._rows.get(k)
                if row:
                    self._rows[k] = row[:position] + (value,) + row[position + 1:]
            self.version += 1

    def set_flags(self, changes: Dict[str, int]) -> None:
        with self.lock:
            self._dirty_keys(changes)
        self._change(13, changes)

    def put_usage(self, rows: Iterable[tuple]) -> None:
        rows = list(rows)
        with self.lock:
            self._dirty_keys([row[0] for row in rows])
            for k, cpu, rss, util in rows:
                self._usage[k] = (cpu, rss, util, True)
            self.version += 1

    def mark_checked(self, keys: Iterable[str]) -> None:
        with self.lock:
            for k in keys:
                self._usage.setdefault(k, (None, None, None, True))
            self.version += 1

    def unchecked(self, low: Optional[int], high: int) -> List[Tuple[str, int, int, int]]:
        with self.lock:
            found = [(r[0], r[2], r[6], r[7]) for r in self._rows.values()
                     if r[6] is not None and r[7] is not None and r[7] >= (low or 0)
                     and r[7] < OPEN_STOP and r[6] < high and r[7] > r[6]
                     and r[0] not in self._usage]
        return sorted(found, key=lambda item: item[3])

    def clear(self) -> None:
        with self.lock:
            self._clear()
            self._forget()
            self.version += 1


# Where a home directory on one of these makes the cache slow (measured on virtiofs: 6 MB/s
# of writes and a round trip per page read, so a cold load spent most of its time there).
NETWORK_FILESYSTEMS = frozenset((
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "virtiofs", "9p", "lustre", "gpfs", "beegfs",
    "ceph", "glusterfs", "afs", "panfs", "pvfs2", "orangefs", "wekafs", "davfs", "sshfs"))
LOCAL_BASE = "/var/tmp"  # node-local disk by convention, unlike /tmp (often RAM)
BASE_OWNERS = (0,)  # who may own LOCAL_BASE


def _unescape(text: str) -> str:
    return text.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n") \
        .replace("\\134", "\\")


def filesystem_type(path: str, mountinfo: str = "/proc/self/mountinfo") -> Optional[str]:
    """The type of the filesystem holding ``path`` (Linux), or None if it cannot be told."""
    path = os.path.realpath(path)
    best, found = -1, None  # type: int, Optional[str]
    try:
        with open(mountinfo, encoding="utf-8", errors="replace") as lines:
            for line in lines:
                left, _, right = line.partition(" - ")
                fields, kinds = left.split(), right.split()
                if len(fields) < 5 or not kinds:
                    continue
                point = _unescape(fields[4])
                inside = path == point or path.startswith(point.rstrip("/") + "/")
                if inside and len(point) >= best:  # later lines mount over earlier ones
                    best, found = len(point), kinds[0]
    except OSError:
        return None
    return found


def is_network(kind: Optional[str]) -> bool:
    return kind is not None and (kind in NETWORK_FILESYSTEMS or kind.startswith("fuse."))


def _local_cache_dir(base: Optional[str] = None) -> Optional[str]:
    """An owner-only ``slurm-usage-<uid>`` directory on node-local disk, or None.

    Only under a root-owned sticky (or not world-writable) base on a local filesystem,
    so no one else can pre-create, rename or replace it; never a subdirectory of
    something another user made."""
    base = base or LOCAL_BASE
    try:
        found = os.lstat(base)
        if (not stat.S_ISDIR(found.st_mode) or found.st_uid not in BASE_OWNERS
                or (found.st_mode & 0o002 and not found.st_mode & stat.S_ISVTX)
                or is_network(filesystem_type(base)) or filesystem_type(base) is None):
            return None
        directory = os.path.join(base, "slurm-usage-%d" % os.geteuid())
        if not os.path.lexists(directory):
            os.mkdir(directory, 0o700)
            os.chmod(directory, 0o700)
        return directory if _is_private_dir(directory) else None
    except OSError:
        return None


def home_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME", "")
    if not os.path.isabs(base):  # the XDG specification says to ignore relative values
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "slurm-usage")


def default_cache_dir() -> str:
    """``$XDG_CACHE_HOME/slurm-usage``, or ``~/.cache/slurm-usage``; but when XDG_CACHE_HOME
    is not set and that would be on a network filesystem, a private directory on
    node-local disk (``/var/tmp/slurm-usage-<uid>``), if one can be had safely. Each node
    then keeps its own cache, which is disposable anyway."""
    chosen = home_cache_dir()
    if not os.path.isabs(os.environ.get("XDG_CACHE_HOME", "")):
        probe = chosen
        while not os.path.exists(probe) and os.path.dirname(probe) != probe:
            probe = os.path.dirname(probe)
        if is_network(filesystem_type(probe)):
            return _local_cache_dir() or chosen
    return chosen


def cache_name(*identity: object) -> str:
    digest = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:20]
    return "history-v%d-%s.sqlite" % (SCHEMA, digest)


SUBDIR = "slurm-usage"


def _is_private_dir(path: str) -> bool:
    found = os.lstat(path)
    return (stat.S_ISDIR(found.st_mode) and found.st_uid == os.geteuid()
            and not found.st_mode & 0o077)


def private_dir(directory: str) -> str:
    """An owner-only directory for the cache, or raise OSError.

    The mode of a directory that already exists is never changed (it may be shared,
    group-owned or setgid). Use it if it is already private to you; otherwise use a
    ``slurm-usage`` subdirectory created by this tool, which must itself be private.
    """
    for candidate in (directory, os.path.join(directory, SUBDIR)):
        if not os.path.lexists(candidate):
            parent = os.path.dirname(candidate.rstrip("/")) or "."
            os.makedirs(parent, exist_ok=True)
            os.mkdir(candidate, 0o700)  # new and ours: only the umask can loosen it
            os.chmod(candidate, 0o700)
        if _is_private_dir(candidate):
            return candidate
    raise OSError("%s is not private to you; not using it" % os.path.join(directory, SUBDIR))


def _private_file(directory: str, name: str) -> str:
    """Create or check an owner-only file in a private directory, or raise OSError."""
    directory = private_dir(directory)
    path = os.path.join(directory, name)
    handle = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        found = os.fstat(handle)
        if not stat.S_ISREG(found.st_mode) or found.st_uid != os.geteuid() or found.st_nlink != 1:
            raise OSError("%s is not a file owned by you" % path)
        if stat.S_IMODE(found.st_mode) != 0o600:
            os.fchmod(handle, 0o600)
            if os.fstat(handle).st_mode & 0o077:
                raise OSError("%s cannot be made private" % path)
    finally:
        os.close(handle)
    return path


def _remove_old_versions(directory: str, name: str) -> None:
    """Caches of this identity written by another schema version are disposable."""
    suffix = name.split("-", 2)[-1]
    for other in os.listdir(directory):
        if (other != name and other.startswith("history-v") and other.endswith("-" + suffix)
                and other.split("-", 2)[-1] == suffix):
            try:
                os.unlink(os.path.join(directory, other))
            except OSError:
                pass


def _remove_home_copies(name: str) -> None:
    """The cache moved to node-local disk: drop this identity's files from home."""
    home = home_cache_dir()
    try:
        if _is_private_dir(home):
            _remove_old_versions(home, name)
            os.unlink(os.path.join(home, name))
    except OSError:
        pass


def open_store(directory: Optional[str], name: str, persist: bool = True) -> _Base:
    """The best store available. ``notes`` says what had to be given up."""
    if sqlite3 is None:
        store = MemoryStore()  # type: _Base
        store.notes.append(Note("cache_memory", "Python has no sqlite3: history is kept in "
                                                "memory and reloaded each run"))
        return store
    problem = ""
    if persist:
        try:
            path = _private_file(directory or default_cache_dir(), name)
            _remove_old_versions(os.path.dirname(path), name)
            if directory is None and not path.startswith(home_cache_dir() + os.sep):
                _remove_home_copies(name)
            try:
                return SqliteStore(path)
            except sqlite3.OperationalError as error:  # locked or unwritable: not ours to delete
                problem = str(error)
            except sqlite3.DatabaseError:  # corrupt, and by definition disposable
                os.unlink(path)
                store = SqliteStore(_private_file(directory or default_cache_dir(), name))
                store.notes.append(Note("cache_rebuilt", "the history cache was unreadable "
                                                         "and has been rebuilt"))
                return store
        except (OSError, sqlite3.Error) as error:
            problem = str(error)
    store = SqliteStore()
    if problem:
        store.notes.append(Note("cache_memory", "history cache unavailable (%s): using memory"
                                % problem))
    return store
