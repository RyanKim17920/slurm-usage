"""Reads on a connection per thread, the rollup built between the pieces of a first load,
run-type medians from one shared read, and job-name-family trends from the family
aggregates: each answers exactly what the one-at-a-time way does. Invented data."""

import os
import sqlite3
import threading
import unittest

from tests.test_history_support import make_history, temp_dir
from tests.test_perf_aggregates import NOW, invented_jobs, windows
from tests.test_history_support import FakeSlurm

from slurm_usage.history import Filter, Window
from slurm_usage.history.runtypes import run_types
from slurm_usage.history.store import OPEN_STOP, SqliteStore

H, DAY = 3600, 86400


def loaded(case, jobs, between=True, parallel=True):
    fake = FakeSlurm(jobs, now=NOW, live=[j["JobIDRaw"] for j in jobs if j["End"] is None])
    saved, SqliteStore.parallel_reads = SqliteStore.parallel_reads, parallel
    try:
        history = make_history(fake, persist=True, cache_dir=temp_dir(case))
        history.info()  # the store is opened here
    finally:
        SqliteStore.parallel_reads = saved
    history.ROLLUP_BETWEEN = between
    history.refresh()
    return fake, history


def tables(store, names):
    return {name: store._select("SELECT * FROM %s ORDER BY 1,2,3" % name) for name in names}


class RollupBetweenPieces(unittest.TestCase):
    def test_same_rollup_and_aggregates_as_building_afterwards(self):
        for seed in (31, 32):
            jobs = invented_jobs(seed, count=600)
            _, before = loaded(self, jobs, between=False)
            _, during = loaded(self, jobs, between=True)
            # the load built most of it already: what is left is the oldest part
            self.assertIsNotNone(during._store.get_meta("rollup_until"))
            left, whole = during.build_rollup(), before.build_rollup()
            self.assertGreater(whole, 0)
            self.assertLess(left, whole)
            names = ("rollup", "agg_hour", "agg_start", "agg_fhour", "agg_fstart")
            a, b = tables(before._store, names), tables(during._store, names)
            fams = [dict(h._store._select("SELECT id,v FROM fams")) for h in (before, during)]
            for name in names:
                if name.startswith("agg_f"):  # family ids may be interned in another order
                    a[name] = sorted(r[:2] + (fams[0][r[2]],) + r[3:] for r in a[name])
                    b[name] = sorted(r[:2] + (fams[1][r[2]],) + r[3:] for r in b[name])
                self.assertEqual(a[name], b[name], "%s seed %d" % (name, seed))
            self.assertIsNone(during._store.get_meta("dirty") or None)
            for r1, r2 in zip(before.overview(gpu_capacity=64).ranges,
                              during.overview(gpu_capacity=64).ranges):
                self.assertEqual(r1.total.gpu_seconds, r2.total.gpu_seconds)

    def test_unfinished_rows_read_again_in_one_load_are_not_rewritten(self):
        from tests.test_history_support import job
        jobs = [job(1, start=NOW - 3 * DAY, end=None, state="RUNNING", gpus=2),
                job(2, start=NOW - 2 * DAY, end=NOW - DAY, gpus=1)]
        _, history = loaded(self, jobs)
        history.build_rollup()
        store = history._store
        stored = store._select("SELECT * FROM jobs WHERE k=(SELECT k FROM jobs WHERE stop=?)",
                               (OPEN_STOP,))
        self.assertEqual(len(stored), 1)
        dims, sets = store.dims(), store.nodesets()
        records = [_record(row, dims, sets) for row in stored]
        generation = stored[0][17]
        self.assertIsNone(store.get_meta("dirty"))
        # read again by the same pass, as each piece going back reads what still runs:
        # nothing to write, so the rollup built so far stays good
        store.put(records, generation)
        self.assertIsNone(store.get_meta("dirty"))
        # read by another pass, it is written again (and its span to be rebuilt)
        store.put(records, generation + 1)
        self.assertIsNotNone(store.get_meta("dirty"))


def _record(row, dims, sets):
    from slurm_usage.history.records import Record
    fields = dict(key=row[0], job=row[1], label=row[2], name=row[3], roles=dims[row[4]],
                  nodes=sets[row[5]], submit=row[6], eligible=row[7], start=row[8],
                  end=None if row[9] == OPEN_STOP else row[9], suspended=row[10], cpus=row[11],
                  gpus=row[12], mem=row[13], limit=row[14], restarts=row[15], exit=row[18])
    return Record(**{k: v for k, v in fields.items() if k in Record._fields})


class ParallelReads(unittest.TestCase):
    def test_threads_read_on_their_own_connections_and_agree(self):
        _, history = loaded(self, invented_jobs(34, count=500))
        history.build_rollup()
        store = history._store
        self.assertTrue(store._parallel)
        as_of = history._as_of()
        cases = [(w, f, by) for w in windows(34, as_of)[:6]
                 for f in (Filter(), Filter(user="carol"))
                 for by in ("gpu_count", "account", "name_family", "outcome")]

        def answer(case):
            w, f, by = case
            return sorted((t.key, t.stats.attempts, t.stats.gpu_seconds, t.median_runtime,
                           t.median_wait) for t in run_types(store, as_of, w, f, by))
        serial = [answer(c) for c in cases]
        found, seen, errors = {}, set(), []

        def work(part):
            try:
                seen.add(id(store._conn()))
                for at in part:
                    found[at] = answer(cases[at])
            except Exception as error:  # reported below
                errors.append(error)
        threads = [threading.Thread(target=work, args=(range(i, len(cases), 3),))
                   for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(seen), 3)  # three connections
        self.assertNotIn(id(store._db), seen)  # none of them the writer's
        self.assertEqual([found[i] for i in range(len(cases))], serial)

    def test_a_read_sees_one_state_while_another_thread_writes(self):
        _, history = loaded(self, invented_jobs(35, count=200))
        store = history._store
        before = store.count()
        inside, wrote = [], threading.Event()

        def writer():
            store.set_meta("probe", 1)  # waits for the reader's transaction to end
            wrote.set()
        with store.reading():
            inside.append(store.count())
            t = threading.Thread(target=writer)
            t.start()
            t.join(0.3)
            self.assertFalse(wrote.is_set())  # held off while the read lasts
            inside.append(store.count())
        t.join(5)
        self.assertTrue(wrote.is_set())
        self.assertEqual(inside, [before, before])
        self.assertEqual(store.get_meta("probe"), 1)

    def test_a_write_outlasting_the_timeout_waits_for_reads_of_this_process(self):
        _, history = loaded(self, invented_jobs(42, count=100))
        store = history._store
        store._db.execute("PRAGMA busy_timeout=50")  # shorter than the read below
        started, release = threading.Event(), threading.Event()

        def reader():
            with store.reading():
                store.count()
                started.set()
                release.wait(5)
        t = threading.Thread(target=reader)
        t.start()
        started.wait(5)
        threading.Timer(0.4, release.set).start()
        store.set_meta("probe", 3)  # retried until the read is over, not failed
        t.join(5)
        self.assertEqual(store.get_meta("probe"), 3)

    def test_a_write_from_inside_a_read_does_not_wait_for_itself(self):
        _, history = loaded(self, invented_jobs(36, count=100))
        store = history._store
        with store.reading():
            store.count()
            store.set_meta("probe", 2)  # ends this thread's read first
            self.assertEqual(store.get_meta("probe"), 2)

    def test_serial_mode_reads_on_the_one_connection(self):
        saved, SqliteStore.parallel_reads = SqliteStore.parallel_reads, False  # the default
        self.addCleanup(setattr, SqliteStore, "parallel_reads", saved)  # (--once sets it)
        store = SqliteStore(os.path.join(temp_dir(self), "serial.sqlite"))
        self.assertFalse(store._parallel)
        self.assertIs(store._conn(), store._db)
        self.assertIs(store._guard(), store.lock)
        self.assertFalse(SqliteStore()._parallel)  # in memory: one connection
        store.close()
        _, history = loaded(self, invented_jobs(37, count=200), parallel=False)
        self.assertFalse(history._store._parallel)
        as_of = history._as_of()
        history.build_rollup()
        _, other = loaded(self, invented_jobs(37, count=200))
        other.build_rollup()
        for by in ("gpu_count", "name_family"):
            self.assertEqual(
                [(t.key, t.stats.gpu_seconds, t.median_runtime)
                 for t in run_types(history._store, as_of, Window(None, None), Filter(), by)],
                [(t.key, t.stats.gpu_seconds, t.median_runtime)
                 for t in run_types(other._store, as_of, Window(None, None), Filter(), by)])

    def test_connections_of_finished_threads_are_closed(self):
        _, history = loaded(self, invented_jobs(38, count=50))
        store = history._store
        made = []
        t = threading.Thread(target=lambda: made.append(store._conn()) or store.count())
        t.start()
        t.join()
        store.count()  # a new thread's connection is made on first use ...
        t2 = threading.Thread(target=store.count)
        t2.start()
        t2.join()  # ... and the finished thread's one closed then
        with self.assertRaises(sqlite3.ProgrammingError):
            made[0].execute("SELECT 1")
        store.close()


class SharedMedians(unittest.TestCase):
    def test_every_kind_from_one_read_equals_the_exact_pass(self):
        _, history = loaded(self, invented_jobs(39, count=500))
        history.build_rollup()
        store, as_of = history._store, history._as_of()
        reads = []
        original = store._select

        def counting(sql, args=()):
            if "j.start-j.eligible" in sql:
                reads.append(sql)
            return original(sql, args)
        store._select = counting
        for w in windows(39, as_of)[:6]:
            for f in (Filter(), Filter(user="alice"), Filter(outcome=["completed"])):
                del reads[:]
                for by in ("gpu_count", "account", "qos", "partition", "outcome",
                           "name_family"):
                    fast = {t.key: (t.median_runtime, t.median_wait)
                            for t in run_types(store, as_of, w, f, by)}
                    exact = {t.key: (t.median_runtime, t.median_wait)
                             for t in run_types(store, as_of, w, f, by, exact=True)}
                    self.assertEqual(fast, exact, "%s %s %s" % (w, f, by))
                self.assertLessEqual(len(reads), 1, "one read of the rows for every kind")


class FamilyTrends(unittest.TestCase):
    def test_family_series_from_the_aggregates_equal_the_rows(self):
        from slurm_usage.ui import breakdown, data as hd
        jobs = invented_jobs(40, count=500)
        for found in jobs:  # suspension is spread over a run: the plain overlap is exact
            found["Suspended"] = "00:00:00"
        _, history = loaded(self, jobs)
        history.build_rollup()
        data = hd.HistoryData(history, clock=lambda: NOW)
        for name in ("day", "week", "month", "year"):
            for f in (None, hd.where(user="bob")):
                edges, got = data.kind_trends(name, f, ("name_family",))
                _, gpus = data.kind_trends(name, f, ("gpu_count",))
                fam = got["name_family"]
                win = hd.window(name, NOW)
                page = history.jobs(win, f, by="attempt", sort="start", limit=10 ** 6,
                                    count=False)
                rows = [hd.from_history_row(r) for r in page.rows] \
                    if hasattr(hd, "from_history_row") else page.rows
                ref = breakdown.held_per_bin(rows, edges, NOW,
                                             lambda r: hd._key(r, "name_family"))
                for key in set(ref) | set(fam):
                    zero = [0] * (len(edges) - 1)
                    self.assertEqual(list(fam.get(key, zero)),
                                     [int(round(v)) for v in ref.get(key, zero)],
                                     "%s %s" % (name, key))
                for at in range(len(edges) - 1):  # every bin sums to the GPU series
                    self.assertEqual(sum(v[at] for v in fam.values()),
                                     sum(v[at] for v in gpus["gpu_count"].values()))

    def test_without_family_aggregates_none_is_answered(self):
        _, history = loaded(self, invented_jobs(41, count=50))
        history.build_rollup()
        history._store.set_meta("agg_fam", -1)
        self.assertIsNone(history.kind_series(Window(NOW - DAY, None), None,
                                              kinds=("name_family",)))


if __name__ == "__main__":
    unittest.main()
