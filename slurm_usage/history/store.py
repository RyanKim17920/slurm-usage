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
CREATE INDEX IF NOT EXISTS jobs_long ON jobs(start) WHERE stop - start > 172800;
CREATE INDEX IF NOT EXISTS jobs_odd ON jobs(start) WHERE stop < start;
CREATE INDEX IF NOT EXISTS jobs_dim_cov ON jobs(dim, start, stop, gpus, cpus, susp, job);
CREATE TABLE IF NOT EXISTS agg_hour(
  hour INTEGER, dim INTEGER, gpus INTEGER, gpu INTEGER, cpu INTEGER, mem_mib INTEGER, mem_rest INTEGER,
  secs INTEGER, cpu_cov INTEGER, cpu_used REAL, mcov_mib INTEGER, mcov_rest INTEGER,
  mem_used REAL, gpu_cov INTEGER, gpu_used REAL,
  f_any INTEGER, l_any INTEGER, f_gpu INTEGER, l_gpu INTEGER, f_cpu INTEGER, l_cpu INTEGER,
  f_mem INTEGER, l_mem INTEGER, PRIMARY KEY(hour, dim, gpus)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS agg_start(
  hour INTEGER, dim INTEGER, gpus INTEGER, attempts INTEGER, cpu_known INTEGER, mem_known INTEGER,
  gpu_known INTEGER, susp INTEGER, wait_count INTEGER, wait_total INTEGER, wait_max INTEGER,
  limit_count INTEGER, limit_used INTEGER, limit_requested INTEGER, over100 INTEGER,
  unchecked INTEGER, PRIMARY KEY(hour, dim, gpus)) WITHOUT ROWID;
"""
# Rows longer than this are found through the partial index jobs_long; all others overlap
# a moment only if they started at most this long before it.
LONG = 172800
AGG_VERSION = 3  # of the exact aggregates (agg_hour, agg_start): a change rebuilds them
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

    def _write(self, work, quiet: bool = False) -> None:
        with self.lock:
            try:
                # One transaction from the first read on: otherwise each SELECT takes and
                # drops its own lock, a round trip apiece on a network filesystem.
                if not self._db.in_transaction:
                    self._db.execute("BEGIN IMMEDIATE")
                work(self._db)
                if not quiet:  # a generation that later runs can compare with
                    self._db.execute("INSERT OR REPLACE INTO meta VALUES('gen', COALESCE("
                                     "(SELECT v FROM meta WHERE k='gen'), 0) + 1)")
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

    def generation(self) -> Optional[int]:
        """Bumped by every change to the data, by any process: equal generations, equal
        answers (for a fixed window)."""
        return self.get_meta("gen", 0)

    # bookkeeping that never changes an answer: writing it does not change ``version``
    QUIET_META = frozenset(("lookback_at",))

    def set_meta(self, key: str, value: Any) -> None:
        text = json.dumps(value)
        with self.lock:
            found = self._select("SELECT v FROM meta WHERE k=?", (key,))
            if found and found[0][0] == text:
                return  # unchanged: nothing to write, nothing to recompute
            version = self.version
            quiet = key in self.QUIET_META
            self._write(lambda db: db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)",
                                              (key, text)), quiet)
            if quiet:
                self.version = version

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
                       dirty: Optional[Tuple[int, int]], agg: Optional[tuple] = None) -> None:
        """Replace hours [low, high) and record what is still dirty. ``agg`` is
        (hour rows, start rows) for the exact aggregates (see aggregate.py)."""
        def work(db):
            db.execute("DELETE FROM rollup WHERE hour>=? AND hour<?", (low, high))
            # byte-seconds outgrow SQLite's 64-bit sums, so memory is kept as MiB + rest
            db.executemany("INSERT INTO rollup VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           [r[:4] + (r[4] >> 20, r[4] & _MIB_MASK) + tuple(r[5:]) for r in rows])
            if agg is not None:
                hours, starts = agg
                db.execute("DELETE FROM agg_hour WHERE hour>=? AND hour<?", (low, high))
                db.execute("DELETE FROM agg_start WHERE hour>=? AND hour<?", (low, high))
                # (hour, dim, gpus, gpu, cpu, mem, secs, cpu_cov, cpu_used, mem_cov, ...)
                db.executemany("INSERT INTO agg_hour VALUES(%s)" % ",".join("?" * 23),
                               [r[:5] + (r[5] >> 20, r[5] & _MIB_MASK) + r[6:9]
                                + (r[9] >> 20, r[9] & _MIB_MASK) + r[10:] for r in hours])
                db.executemany("INSERT INTO agg_start VALUES(%s)" % ",".join("?" * 16), starts)
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

    def rollup_segments(self, bounds: Sequence[int], high: int) -> List[tuple]:
        """rollup_sum()'s columns, a row count and the first and last hour per (dim,
        segment) for hours before
        ``high``: segment i
        holds the hours from bounds[i] (descending) up to the previous bound, or ``high``;
        segment len(bounds) holds everything older."""
        case = "CASE %s ELSE %d END" % (" ".join("WHEN hour>=%d THEN %d" % (int(b), i)
                                                 for i, b in enumerate(bounds)), len(bounds))
        found = self._select(
            "SELECT dim,%s,sum(gpu),sum(cpu),sum(mem_mib),sum(mem_rest),sum(attempts),"
            "sum(cpu_cov),sum(cpu_used),total(mem_cov),total(mem_used),sum(gpu_cov),"
            "sum(gpu_used),count(*),min(hour),max(hour) FROM rollup WHERE hour<? GROUP BY 1,2"
            % case, (high,))
        return [r[:4] + (((r[4] or 0) << 20) + (r[5] or 0),) + tuple(r[6:]) for r in found]

    def rollup_series(self, low: int, high: int, step: int) -> List[Tuple[int, int, int]]:
        """(bins back from ``high``, gpu, cpu) for hours in [low, high); bin 0 ends at high."""
        return self._select(
            "SELECT (?-1-hour)/?,sum(gpu),sum(cpu) FROM rollup WHERE hour>=? AND hour<? "
            "GROUP BY 1", (high, step, low, high))

    def rollup_activity(self) -> List[Tuple[int, int, int]]:
        """(dim, first hour, last hour) with any usage."""
        return self._select("SELECT dim,min(hour),max(hour) FROM rollup GROUP BY dim")

    # -- exact aggregates of finished allocations (aggregate.py's fast path) ---------

    has_agg = True

    @staticmethod
    def _dims_sql(dim_ids: Optional[Set[int]], column: str = "dim") -> str:
        if dim_ids is None:
            return ""
        return " AND %s IN (%s)" % (column, ",".join(str(int(i)) for i in dim_ids))

    def agg_reset(self) -> None:
        """Empty the exact aggregates, made again in the current layout."""
        def work(db):
            db.execute("DROP TABLE IF EXISTS agg_hour")
            db.execute("DROP TABLE IF EXISTS agg_start")
            for statement in _TABLES.split(";"):
                if "agg_" in statement:
                    db.execute(statement)

        self._write(work)

    def agg_busy(self, low: int, high: int, dim_ids: Optional[Set[int]], metric: Optional[str]
                 ) -> List[Tuple[int, int, int, int]]:
        """(hour, dim, first, last): when finished allocations (holding ``metric``, if
        given) were first and last held within each hour of [low, high)."""
        column = metric or "any"
        return self._select(
            "SELECT hour,dim,min(f_%s),max(l_%s) FROM agg_hour WHERE hour>=? AND hour<? AND "
            "f_%s IS NOT NULL%s GROUP BY hour,dim ORDER BY hour" % (
                column, column, column, self._dims_sql(dim_ids)), (low, high))

    def agg_sums(self, low: int, high: int, dim_ids: Optional[Set[int]],
                 by_gpus: bool = False) -> Tuple[Dict[Any, tuple], Dict[Any, tuple]]:
        """Per dim (or (dim, GPUs per allocation, None if unknown)) over hours [low, high):
        (gpu, cpu, mem, secs, cpu_cov, cpu_used, mem_cov, mem_used, gpu_cov, gpu_used) of
        finished allocations, and the per-attempt counts of those that started then
        (agg_start's columns)."""
        where = self._dims_sql(dim_ids)
        key = "dim,gpus" if by_gpus else "dim,NULL"

        def name(r: tuple) -> Any:
            return (r[0], None if r[1] == -1 else r[1]) if by_gpus else r[0]
        hours = {}
        for r in self._select(
                "SELECT " + key + ",sum(gpu),sum(cpu),sum(mem_mib),sum(mem_rest),sum(secs),sum(cpu_cov),"
                "total(cpu_used),sum(mcov_mib),sum(mcov_rest),total(mem_used),sum(gpu_cov),"
                "total(gpu_used) FROM agg_hour WHERE hour>=? AND hour<?%s GROUP BY 1,2" % where,
                (low, high)):
            k, r = name(r), r[1:]
            hours[k] = (r[1] or 0, r[2] or 0, ((r[3] or 0) << 20) + (r[4] or 0), r[5] or 0,
                        r[6] or 0, r[7], ((r[8] or 0) << 20) + (r[9] or 0), r[10],
                        r[11] or 0, r[12])
        starts = {}
        for r in self._select(
                "SELECT " + key + ",sum(attempts),sum(cpu_known),sum(mem_known),sum(gpu_known),sum(susp),"
                "sum(wait_count),sum(wait_total),max(wait_max),sum(limit_count),sum(limit_used),"
                "sum(limit_requested),sum(over100),sum(unchecked) FROM agg_start "
                "WHERE hour>=? AND hour<?%s GROUP BY 1,2" % where, (low, high)):
            starts[name(r)] = tuple(r[2:])
        return hours, starts

    def agg_series(self, low: int, high: int, dim_ids: Optional[Set[int]],
                   mode: Optional[str]) -> List[tuple]:
        """(hour, key, gpu, cpu, mem) per hour in [low, high); key is None, the dim
        (mode "dim") or (dim, GPUs per allocation or None) (mode "dimgpus")."""
        where = self._dims_sql(dim_ids)
        if mode == "dimgpus":
            return [(r[0], (r[1], None if r[2] == -1 else r[2]), r[3], r[4], (r[5] << 20) + r[6])
                    for r in self._select(
                        "SELECT hour,dim,gpus,sum(gpu),sum(cpu),sum(mem_mib),sum(mem_rest) FROM "
                        "agg_hour WHERE hour>=? AND hour<?%s GROUP BY hour,dim,gpus" % where,
                        (low, high))]
        if mode == "dim":
            return [(r[0], r[1], r[2], r[3], (r[4] << 20) + r[5]) for r in self._select(
                "SELECT hour,dim,sum(gpu),sum(cpu),sum(mem_mib),sum(mem_rest) FROM agg_hour "
                "WHERE hour>=? AND hour<?%s GROUP BY hour,dim" % where, (low, high))]
        return [(r[0], None, r[1] or 0, r[2] or 0, ((r[3] or 0) << 20) + (r[4] or 0))
                for r in self._select(
                    "SELECT hour,sum(gpu),sum(cpu),sum(mem_mib),sum(mem_rest) FROM agg_hour "
                    "WHERE hour>=? AND hour<?%s GROUP BY hour" % where, (low, high))]

    def agg_binned(self, low: int, high: int, dim_ids: Optional[Set[int]],
                   bounds: Sequence[int]) -> List[tuple]:
        """(bin, dim, gpus or None, GPU-seconds) of finished allocations over hours [low,
        high), bin i holding the hours from low (or bounds[i]) to the next bound."""
        cuts = sorted(set(b for b in bounds if low < b < high))
        case = "CASE %s ELSE 0 END" % " ".join(
            "WHEN hour>=%d THEN %d" % (b, len(cuts) - at) for at, b in
            enumerate(reversed(cuts))) if cuts else "0"
        return [(r[0], r[1], None if r[2] == -1 else r[2], r[3]) for r in self._select(
            "SELECT %s,dim,gpus,sum(gpu) FROM agg_hour WHERE hour>=? AND hour<?%s "
            "GROUP BY 1,2,3" % (case, self._dims_sql(dim_ids)), (low, high))]

    def agg_waits(self, low: int, high: int, dim_ids: Optional[Set[int]],
                  keyed: bool = False) -> List[Any]:
        """Queue waits of finished allocations that started and ended in [low, high);
        keyed: (dim, gpus, wait)."""
        found = self._select(
            "SELECT dim,gpus,start-eligible FROM jobs WHERE start>=? AND start<? AND stop<? "
            "AND stop>=start AND eligible<=start%s" % self._dims_sql(dim_ids), (low, high, high))
        return found if keyed else [r[2] for r in found]

    def runtimes(self, low: Optional[int], high: int, start: int, end: int,
                 dim_ids: Optional[Set[int]]) -> List[Tuple[int, Optional[int], int]]:
        """(dim, gpus, run time less suspension, never negative) of the finished rows a
        window counts (as Selection.parts does)."""
        rs = "max(stop,start)"
        p, q = "max(start,%d)" % start, "min(%s,%d)" % (rs, end)
        here = "start<%d" % int(high) + ("" if low is None else " AND start>=%d" % int(low))
        where = "start<%d AND stop<%d%s AND (%s>%s OR (%s=start AND %s))" % (
            int(high), OPEN_STOP, "" if low is None else " AND stop>=%d" % int(low), q, p, rs,
            here)
        return self._select("SELECT dim,gpus,max(0,stop-start-susp) FROM jobs WHERE %s%s" % (
            where, self._dims_sql(dim_ids)))

    def type_medians(self, low: Optional[int], high: int, start: int, end: int, as_of: int,
                     finished: Optional[Set[int]], running: Optional[Set[int]],
                     skip: Sequence[str], keys: Optional[Dict[int, Tuple[str, str]]]
                     ) -> Tuple[Dict[Any, int], Dict[Any, int], Dict[Any, str]]:
        """Lower medians per kind, and the name of the earliest row, of the run times (finished rows) and queue waits (rows
        that started in the window) of the rows a window counts, less ``skip``. Kinds are
        GPUs per allocation (-1 unknown) or, with ``keys``, keys[dim] (finished, unfinished)."""
        with self.lock:
            if keys is not None:
                self._db.execute("CREATE TEMP TABLE IF NOT EXISTS kinds(dim INTEGER PRIMARY KEY, "
                                 "kind TEXT, open TEXT)")
                self._db.execute("DELETE FROM temp.kinds")
                self._db.executemany("INSERT INTO temp.kinds VALUES(?,?,?)",
                                     [(d,) + tuple(k) for d, k in keys.items()])
                self._db.commit()
            rs = "(CASE WHEN j.stop=%d THEN max(%d,j.start) ELSE max(j.stop,j.start) END)" % (
                OPEN_STOP, int(as_of))
            p, q = "max(j.start,%d)" % int(start), "min(%s,%d)" % (rs, int(end))
            here = "j.start<%d" % int(high) + ("" if low is None else " AND j.start>=%d" % int(low))
            fin = "j.stop<%d" % OPEN_STOP + ("" if finished is None else
                                            self._dims_sql(finished, "j.dim"))
            run = "j.stop=%d" % OPEN_STOP + ("" if running is None else
                                            self._dims_sql(running, "j.dim"))
            where = ["j.start<%d" % int(high), "((%s) OR (%s))" % (fin, run),
                     "(%s>%s OR (%s=j.start AND %s))" % (q, p, rs, here)]
            if low is not None:
                where.append("j.stop>=%d" % int(low))
            if skip:
                where.append("j.k NOT IN (%s)" % ",".join(
                    "'%s'" % k.replace("'", "''") for k in skip))
            kind = "coalesce(j.gpus,-1)" if keys is None else \
                "(CASE WHEN j.stop=%d THEN t.open ELSE t.kind END)" % OPEN_STOP
            join = "" if keys is None else " JOIN temp.kinds t ON t.dim=j.dim"
            base = "FROM jobs j%s WHERE %s" % (join, " AND ".join(where))

            found = self._select(
                "SELECT %s,CASE WHEN j.stop<%d THEN max(0,j.stop-j.start-j.susp) END,"
                "CASE WHEN %s AND j.eligible<=j.start THEN j.start-j.eligible END,j.start %s" % (
                    kind, OPEN_STOP, here, base))
            runs, waited = {}, {}  # type: Dict[Any, List[int]], Dict[Any, List[int]]
            first = {}  # type: Dict[Any, int]
            for key, runtime, wait, begun in found:
                if runtime is not None:
                    runs.setdefault(key, []).append(runtime)
                if wait is not None:
                    waited.setdefault(key, []).append(wait)
                if begun < first.get(key, begun + 1):
                    first[key] = begun
            names = {}  # type: Dict[Any, str]
            for key, begun in first.items():
                row = self._select("SELECT j.name %s AND j.start=? AND %s=? LIMIT 1" % (
                    base, kind), (begun, key))
                names[key] = (row[0][0] if row else "") or ""

        def lower_median(values: List[int]) -> int:
            values.sort()
            return values[(len(values) - 1) // 2]
        runtimes = {key: lower_median(values) for key, values in runs.items()}
        waits = {key: lower_median(values) for key, values in waited.items()}
        return runtimes, waits, names

    def scan_edges(self, start: int, low: int, high: int, until: int,
                   dim_ids: Optional[Set[int]] = None, usage: bool = False) -> Iterator[Row]:
        """The rows a window [start, until) needs besides the aggregates of [low, high):
        every allocation that overlaps [start, low) or [high, until), runs across ``high``,
        has no end, or ends before it starts. Unordered; each row once."""
        if dim_ids is not None and not dim_ids:
            return
        check = None
        dims = ""
        if dim_ids is not None:
            if len(dim_ids) <= _MAX_SQL_IDS:
                dims = self._dims_sql(dim_ids, "dim")
            else:
                check = dim_ids
        long_ = "stop - start > %d" % LONG
        pieces = [
            ("start>=? AND start<? AND stop>=?", (start - LONG, low, start)),
            (long_ + " AND start<? AND stop>=?", (low, start)),
            ("start>=? AND start<? AND stop>=?", (max(low, high - LONG), high, high)),
            (long_ + " AND start>=? AND start<? AND stop>=?", (low, high, high)),
            ("start>=? AND start<?", (high, until)),
            ("stop < start AND start>=? AND start<?", (low, high)),
        ]
        keys = " UNION ".join("SELECT k FROM jobs WHERE %s%s" % (sql, dims) for sql, _ in pieces)
        args = [a for _, values in pieces for a in values]
        sql = "SELECT %s%s FROM jobs j%s WHERE j.k IN (%s)" % (
            _COLUMNS, ",u.cpu,u.rss,u.util,u.k IS NOT NULL" if usage else ",NULL,NULL,NULL,0",
            " LEFT JOIN usage u ON u.k=j.k" if usage else "", keys)
        for row in self._select(sql, args):
            if check is None or row[2] in check:
                yield row  # type: ignore

    def scan_started(self, low: int, high: int, dim_ids: Optional[Set[int]] = None
                     ) -> Iterator[Row]:
        """Allocations that started in [low, high), by start, with (name, exit)."""
        if dim_ids is not None and not dim_ids:
            return
        dims = self._dims_sql(dim_ids, "j.dim") if dim_ids is not None and \
            len(dim_ids) <= _MAX_SQL_IDS else ""
        for row in self._select(
                "SELECT %s,NULL,NULL,NULL,0,j.name,j.ex FROM jobs j WHERE j.start>=? AND "
                "j.start<?%s ORDER BY j.start" % (_COLUMNS, dims), (low, high)):
            if dim_ids is None or row[2] in dim_ids:
                yield row  # type: ignore

    def event_rows(self, high: int, end: int, as_of: int, finished: Optional[Set[int]],
                   running: Optional[Set[int]], skip: Sequence[str]) -> str:
        """The FROM/WHERE of the rows events.py counts: started before ``high`` (and
        before ``end`` unless they took no time), finished in ``finished`` dims or
        unfinished in ``running`` dims (None = any), never the keys in ``skip``. Columns:
        start, rs (the end as counted: now while running, never before the start),
        job, dim, gpus, cpus, susp, stop."""
        where = ["start<%d" % int(high)]
        fin = "stop<%d" % OPEN_STOP + ("" if finished is None else self._dims_sql(finished))
        run = "stop=%d" % OPEN_STOP + ("" if running is None else self._dims_sql(running))
        where.append("((%s) OR (%s))" % (fin, run))
        if skip:
            where.append("k NOT IN (%s)" % ",".join("'%s'" % k.replace("'", "''") for k in skip))
        rs = "(CASE WHEN stop=%d THEN max(%d,start) ELSE max(stop,start) END)" % (
            OPEN_STOP, int(as_of))
        where.append("(start<%d OR %s=start)" % (int(end), rs))
        return ("(SELECT start,%s AS rs,job,dim,coalesce(gpus,0) AS gpus,coalesce(cpus,0) "
                "AS cpus,susp,stop,ex FROM jobs WHERE %s)" % (rs, " AND ".join(where)))

    def event_inside(self, rows: str, low: int, failed: Set[int]) -> Tuple[
            Optional[tuple], Optional[tuple], List[tuple], List[tuple]]:
        """From event_rows(), the rows that started at ``low`` or later: the largest
        (gpus, cpus, start, job, dim), the longest finished (run time, start, stop, job,
        gpus), each dim's first (dim, start, job) and the rows in ``failed`` dims (start,
        stop, job, ex), earliest first where it matters."""
        with self.lock:
            top = self._select("SELECT gpus,cpus,start,job,dim FROM %s WHERE start>=? "
                               "ORDER BY gpus DESC,cpus DESC,start LIMIT 1" % rows, (low,))
            longest = self._select(
                "SELECT rs-start-susp AS run,start,stop,job,gpus FROM %s WHERE start>=? AND "
                "stop<%d ORDER BY run DESC,start LIMIT 1" % (rows, OPEN_STOP), (low,))
            firsts = self._select(
                "SELECT dim,min(start),job FROM %s WHERE start>=? GROUP BY dim" % rows, (low,))
            failures = self._select(
                "SELECT start,stop,job,ex FROM %s WHERE start>=? AND stop<%d%s ORDER BY start"
                % (rows, OPEN_STOP, self._dims_sql(failed)), (low,)) if failed else []
        return (top[0] if top else None, longest[0] if longest else None, firsts, failures)

    def event_prior(self, rows: str, low: int, failed: Set[int], chained: Set[int]
                    ) -> Tuple[List[tuple], List[int], List[Tuple[int, int]], Optional[int],
                               List[tuple]]:
        """From event_rows(): per dim before ``low`` (dim, max gpus, max cpus, longest
        finished run); the ends of failed allocations before ``low`` (``failed`` dims);
        the gaps in coverage (start of the next allocation, coverage before it) and the
        coverage at the end; and the rows of jobs that have a finished row in a ``chained``
        dim, by start (start, rs, job, dim, gpus, susp, stop)."""
        with self.lock:
            prior = self._select(
                "SELECT dim,max(gpus),max(cpus),max(CASE WHEN stop<%d THEN rs-start-susp END) "
                "FROM %s WHERE start<? GROUP BY dim" % (OPEN_STOP, rows), (low,))
            ends = [r[0] for r in self._select(
                "SELECT stop FROM %s WHERE start<? AND stop<%d%s" % (
                    rows, OPEN_STOP, self._dims_sql(failed)), (low,))] if failed else []
            gaps, cover = [], None  # type: List[Tuple[int, int]], Optional[int]
            try:
                ordered = self._db.execute("SELECT start,rs FROM %s ORDER BY start" % rows)
                for begun, end in ordered:
                    if cover is None:
                        cover = end
                        continue
                    if begun > cover:
                        gaps.append((begun, cover))
                    if end > cover:
                        cover = end
            except sqlite3.Error as error:
                raise HistoryError("error", "history cache: %s" % error)
            chains = self._select(
                "SELECT start,rs,job,dim,gpus,susp,stop FROM %s WHERE job IN (SELECT job FROM %s "
                "WHERE stop<%d%s) ORDER BY start" % (rows, rows, OPEN_STOP,
                                                     self._dims_sql(chained))) if chained else []
        return prior, ends, gaps, cover, chains

    def repeated_rows(self, low: int, high: int, dim_ids: Optional[Set[int]]
                      ) -> List[Tuple[int, int, int, int, Optional[int]]]:
        """(start, stop, job, dim, gpus) of the finished rows of jobs with several rows
        that started and ended in [low, high)."""
        with self.lock:
            self._repeated_table()
            return self._select(
                "SELECT start,stop,job,dim,gpus FROM jobs WHERE start>=? AND start<? AND stop<? "
                "AND stop>=start AND job IN (SELECT job FROM temp.repeated)%s" % self._dims_sql(
                    dim_ids), (low, high, high))

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

    _REPEATED = "SELECT job FROM temp.repeated"

    def _repeated_table(self) -> None:
        """temp.repeated: repeated_jobs() as a table, for the queries that exclude them."""
        if getattr(self, "_rep_table", None) == self.version:
            return
        jobs = self.repeated_jobs()
        self._db.execute("CREATE TEMP TABLE IF NOT EXISTS repeated(job INTEGER PRIMARY KEY)")
        self._db.execute("DELETE FROM temp.repeated")
        self._db.executemany("INSERT INTO temp.repeated VALUES(?)", [(j,) for j in jobs])
        self._db.commit()  # never leave a transaction open between calls
        self._rep_table = self.version

    def scan_special(self, low: Optional[int], high: int, dim_ids: Optional[Set[int]],
                     extra: bool = False, repeated: bool = True) -> Iterator[Row]:
        """scan()'s rows that have no end or (``repeated``) belong to a job with several
        rows."""
        if dim_ids is not None and not dim_ids:
            return
        with self.lock:
            self._repeated_table()
        where, args = "j.start<?", [high]  # type: str, List[Any]
        if low is not None:
            where += " AND j.stop>=?"
            args.append(low)
        dims = self._dims_sql(dim_ids, "j.dim") if dim_ids is not None and \
            len(dim_ids) <= _MAX_SQL_IDS else ""
        for row in self._select(
                "SELECT %s,NULL,NULL,NULL,0%s FROM jobs j WHERE %s%s AND (j.stop=%d%s)" % (
                    _COLUMNS, ",j.name,j.ex" if extra else "", where, dims, OPEN_STOP,
                    " OR j.job IN (%s)" % self._REPEATED if repeated else ""), args):
            if dim_ids is None or row[2] in dim_ids:
                yield row  # type: ignore

    def rows_of(self, keys: Sequence[str], extra: bool = False) -> List[Row]:
        out = []  # type: List[Row]
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            out += self._select("SELECT %s,NULL,NULL,NULL,0%s FROM jobs j WHERE j.k IN (%s)" % (
                _COLUMNS, ",j.name,j.ex" if extra else "", ",".join("?" * len(part))), part)
        return out

    def top_finished(self, low: Optional[int], high: int, start: int, end: int,
                     dim_ids: Optional[Set[int]], sort: str, descending: bool, count: int,
                     single: bool = True) -> Tuple[int, List[str]]:
        """Finished rows (of jobs with one row, if ``single``), as listing.list_jobs counts
        them: how many
        have usage in the window, and the keys of the first ``count`` by ``sort``
        (gpu, cpu, mem: amount in the window; start), ties by later start, then key."""
        rs = "max(j.stop,j.start)"
        p, q = "max(j.start,%d)" % start, "min(%s,%d)" % (rs, end)
        held = "min(j.susp,%s-j.start)" % rs
        secs = ("(CASE WHEN %(q)s<=%(p)s THEN 0 WHEN %(h)s>0 THEN (%(q)s-j.start)-%(h)s*(%(q)s-"
                "j.start)/(%(rs)s-j.start)-((%(p)s-j.start)-%(h)s*(%(p)s-j.start)/(%(rs)s-j.start)"
                ") ELSE %(q)s-%(p)s END)" % {"q": q, "p": p, "h": held, "rs": rs})
        value = {"gpu": "j.gpus*" + secs, "cpu": "j.cpus*" + secs, "mem": "j.mem*" + secs,
                 "start": "j.start"}[sort]
        where = ["j.start<%d" % int(high), "j.stop<%d" % OPEN_STOP]
        here = "j.start<%d" % int(high)
        if low is not None:
            where.append("j.stop>=%d" % int(low))
            here += " AND j.start>=%d" % int(low)
        where.append("(%s>%s OR (%s=j.start AND %s))" % (q, p, rs, here))
        if single:
            where.append("j.job NOT IN (%s)" % self._REPEATED)
        sql = " AND ".join(where) + (self._dims_sql(dim_ids, "j.dim") if dim_ids is not None
                                     else "")
        with self.lock:
            self._repeated_table()
            total = self._select("SELECT count(*) FROM jobs j WHERE " + sql)[0][0]
            order = "DESC" if descending else "ASC"
            # unknown amounts last either way, as list_jobs sorts them
            keys = [r[0] for r in self._select(
                "SELECT j.k FROM jobs j WHERE %s ORDER BY (%s) IS NULL, (%s) %s, j.start %s, "
                "j.k %s LIMIT ?" % (sql, value, value, order, order, order), (count,))]
        return total, keys

    def usage_of(self, keys: Sequence[str]) -> Dict[str, tuple]:
        """(cpu, rss, util, True) of the rows of ``keys`` whose steps were read."""
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            for k, cpu, rss, util in self._select(
                    "SELECT k,cpu,rss,util FROM usage WHERE k IN (%s)" % ",".join("?" * len(part)),
                    part):
                out[k] = (cpu, rss, util, True)
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
        keys = list(keys)

        def work(db):
            fresh = [k for k in keys if db.execute("INSERT OR IGNORE INTO usage(k) VALUES(?)",
                                                   (k,)).rowcount]
            self._dirty_keys(db, fresh)  # what is measured changes the aggregates

        self._write(work)

    def unchecked(self, low: Optional[int], high: int) -> List[Tuple[str, int, int, int]]:
        """(key, dim, start, end) of finished allocations whose steps were never read."""
        return self._select(
            "SELECT k,dim,start,stop FROM jobs j WHERE stop>=? AND stop<? AND +start<? "
            "AND stop>start AND NOT EXISTS(SELECT 1 FROM usage u WHERE u.k=j.k) ORDER BY stop",
            (0 if low is None else low, OPEN_STOP, high))

    def clear(self) -> None:
        def work(db):
            for table in ("jobs", "usage", "dims", "nodesets", "meta", "rollup", "agg_hour",
                          "agg_start"):
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

    has_agg = False

    def agg_reset(self) -> None:
        pass

    def rollup_replace(self, low: int, high: int, rows: Iterable[tuple],
                       dirty: Optional[Tuple[int, int]], agg: Optional[tuple] = None) -> None:
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

    def generation(self) -> Optional[int]:
        return None  # nothing outlives the process

    def set_meta(self, key: str, value: Any) -> None:
        with self.lock:
            value = json.loads(json.dumps(value))
            if self._meta.get(key, object()) == value:
                return
            self._meta[key] = value
            if key not in SqliteStore.QUIET_META:
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

    def usage_of(self, keys: Sequence[str]) -> Dict[str, tuple]:
        with self.lock:
            return {k: self._usage[k] for k in keys if k in self._usage}

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
