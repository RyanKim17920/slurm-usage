"""The exact aggregates behind fast queries: every answer equals a full pass over the
attempts, before and after incremental changes and across a layout change. Invented data."""

import random
import unittest

from tests.test_history_support import H, T0, FakeSlurm, job, make_history, step

from slurm_usage.history import Filter, Window
from slurm_usage.history import events as events_module
from slurm_usage.history.aggregate import KIND_SERIES, LAZY_FIELDS, build_report
from slurm_usage.history.stats import Stats
from slurm_usage.history.store import AGG_VERSION, LONG

DAY = 86400
NOW = T0 + 40 * DAY
FIELDS = [f for f in Stats.__slots__ if not f.startswith("_")]
USERS = [("alice", 5001), ("bob", 5002), ("carol", 5003), ("dave", 5004)]
STATES = ["COMPLETED"] * 6 + ["FAILED", "TIMEOUT", "CANCELLED", "PREEMPTED", "NODE_FAIL",
                              "OUT_OF_MEMORY"]


def invented_jobs(seed: int, count: int = 400, now: int = NOW):
    rnd = random.Random(seed)
    jobs, job_id = [], 1000
    for _ in range(count):
        job_id += 1
        user, uid = rnd.choice(USERS)
        start = rnd.randrange(T0, now - H)
        length = rnd.choice([0, 30, 600, 3599, 3600, 7200, 5 * H + 17, DAY + 11,
                             LONG + 5 * H, 9 * DAY])
        kind = rnd.random()
        common = dict(user=user, uid=uid, account=rnd.choice(["physics", "chem", "bio"]),
                      qos=rnd.choice(["normal", "high"]), partition=rnd.choice(["gpu", "cpu"]),
                      gpus=rnd.choice([0, 1, 2, 4, 8]), cpus=rnd.choice([1, 4, 16]),
                      mem=rnd.choice(["4G", "16G", "500G"]),
                      submit=start - rnd.randrange(0, 4 * H), limit=rnd.choice([0, 60, 600]),
                      name=rnd.choice(["train_%d" % rnd.randrange(9), "eval", "prep-%d"
                                       % rnd.randrange(50)]),
                      suspended=rnd.choice([0, 0, 0, 0, 120, 7200]))
        if rnd.random() < 0.3:
            common["steps"] = [step("batch", TotalCPU="00:%02d:00" % rnd.randrange(60),
                                    TRESUsageInTot="cpu=00:01:00,mem=%dM,gres/gpuutil=%d" % (
                                        rnd.randrange(1, 9000), rnd.randrange(0, 900)))]
        if kind < 0.05 and start + length > now - DAY:
            jobs.append(job(job_id, start=start, end=None, state="RUNNING", **common))
        elif kind < 0.07:
            jobs.append(job(job_id, start=start, end=start - 50, **common))  # ends before it starts
        elif kind < 0.15:
            # requeued: two or three attempts of one job id
            at = start
            for attempt in range(rnd.choice([2, 3])):
                end = min(at + length, now - 1)
                jobs.append(job(job_id, start=at, end=end, index=job_id * 10 + attempt,
                                state="PREEMPTED" if attempt == 0 else rnd.choice(STATES),
                                **common))
                at = end + rnd.choice([0, 60, 5 * H])
                if at >= now - H:
                    break
        else:
            jobs.append(job(job_id, start=start, end=min(start + length, now - 1),
                            state=rnd.choice(STATES), **common))
    return jobs


def ready(jobs, now=NOW):
    fake = FakeSlurm(jobs, now=now, live=[j["JobIDRaw"] for j in jobs if j["End"] is None])
    history = make_history(fake)
    history.refresh()
    history.build_rollup()
    return fake, history


def windows(seed: int, now: int = NOW):
    rnd = random.Random(seed)
    found = [Window(now - DAY, None), Window(now - 7 * DAY, None), Window(now - 30 * DAY, None),
             Window(None, None), Window(now - 14 * DAY, now - 7 * DAY), Window(T0, now)]
    for _ in range(8):
        a = rnd.randrange(T0 - DAY, now - H)
        found.append(Window(a, rnd.randrange(a + 1, now + 2 * H)))
    return found


class FastEqualsExact(unittest.TestCase):
    """build_report from the aggregates and from every row agree on every field."""

    def assertSame(self, fast, exact, label):
        def same(a, b, where):
            for name in FIELDS:
                x, y = getattr(a, name), getattr(b, name)
                if isinstance(x, float) or isinstance(y, float):
                    self.assertAlmostEqual(x, y, delta=1e-6 * max(1.0, abs(y)),
                                           msg="%s %s %s" % (label, where, name))
                else:
                    self.assertEqual(x, y, "%s %s %s" % (label, where, name))
            self.assertEqual(a.jobs, b.jobs, "%s %s jobs" % (label, where))
        same(fast.total, exact.total, "total")
        self.assertEqual(set(fast.groups), set(exact.groups))
        for spec, group in exact.groups.items():
            self.assertEqual(set(fast.groups[spec]), set(group), (label, spec))
            for key, stats in group.items():
                same(fast.groups[spec][key], stats, "%s=%s" % (spec, key))
        self.assertEqual(fast.values, exact.values, label)
        self.assertEqual({k: v for k, v in fast.keyed_values.items() if any(v)},
                         {k: v for k, v in exact.keyed_values.items() if any(v)}, label)
        self.assertEqual(list(fast.waits), list(exact.waits), label)
        self.assertEqual(fast.efficiency_unchecked, exact.efficiency_unchecked, label)
        self.assertEqual(fast.lanes, exact.lanes, label)
        self.assertEqual((fast.start, fast.end, fast.edges), (exact.start, exact.end, exact.edges))

    def check(self, history, seed, lanes=False):
        store, as_of = history._store, history._as_of()
        rnd = random.Random(seed)
        filters = [Filter(), Filter(user="alice"), Filter(account=["chem", "bio"]),
                   Filter(outcome=["completed", "preempted"]), Filter(qos="high", user="bob")]
        groups = [(("user",), ("user", "account")), (("qos",), ("partition",), ("state",))]
        checked = 0
        for win in windows(seed, as_of):
            for where in filters:
                efficiency = rnd.random() < 0.5
                group = rnd.choice(groups)
                series = rnd.choice([None, "account", "user", KIND_SERIES])
                kwargs = dict(series_by=series, efficiency=efficiency, gpu_tracked=True)
                if lanes:
                    kwargs.update(lanes_by=rnd.choice(["user", "account"]),
                                  lanes_metric=rnd.choice([None, "gpu", "cpu"]),
                                  lane_resolution=rnd.choice([H, 3 * H, 10 * H]))
                fast = build_report(store, as_of, win, where, group, **kwargs)
                exact = build_report(store, as_of, win, where, group, exact=True, **kwargs)
                self.assertSame(fast, exact, "%s %r" % (win, where))
                for name in LAZY_FIELDS:  # computed on first read, exactly
                    self.assertEqual(getattr(fast.total, name), getattr(exact.total, name))
                checked += 1
        return checked

    def test_matches_exact(self):
        for seed in (1, 2, 3):
            _, history = ready(invented_jobs(seed))
            self.assertTrue(history._store.get_meta("agg") == AGG_VERSION)
            self.assertGreater(self.check(history, seed), 50)

    def test_lanes_match_exact_from_an_hour_of_resolution(self):
        _, history = ready(invented_jobs(4))
        self.check(history, 4, lanes=True)

    def test_fast_path_is_used(self):
        _, history = ready(invented_jobs(5))
        store = history._store
        calls = []
        scan = store.scan

        def counting(*args, **kwargs):
            calls.append(args)
            return scan(*args, **kwargs)
        store.scan = counting
        history.query(Window(None, None), group_by=(("user",),))
        self.assertEqual(calls, [])  # no full scan: aggregates and the edges only

    def test_sums_equal_attempt_totals(self):
        fake, history = ready(invented_jobs(6))
        report = history.query(Window(None, None), group_by=(("user",), ("account",)))
        for spec in (("user",), ("account",), ("outcome",)):
            for name in ("gpu_seconds", "cpu_seconds", "mem_byte_seconds", "attempts", "seconds"):
                self.assertEqual(sum(getattr(s, name) for s in report.group(*spec).values()),
                                 getattr(report.total, name), (spec, name))
        self.assertEqual(sum(report.values["gpu"]), report.total.gpu_seconds)
        # the stored hours of finished attempts add up to their allocated GPU-seconds
        store = history._store
        stored = store._select("SELECT sum(gpu) FROM agg_hour")[0][0]
        finished = sum((row[10] or 0) * row[5] for row in [
            (None,) * 5 + (r[1] - r[0],) + (None,) * 4 + (r[2],) for r in store._select(
                "SELECT start,stop,gpus FROM jobs WHERE stop<? AND stop>=start AND susp=0 "
                "AND stop<=?", (2 ** 40, store.get_meta("rollup_until")))])
        suspended = store._select("SELECT count(*) FROM jobs WHERE susp>0")[0][0]
        if not suspended:
            self.assertEqual(stored, finished)
        self.assertGreater(stored, 0)


class Incremental(unittest.TestCase):
    def test_updates_stay_exact(self):
        jobs = invented_jobs(7, now=NOW - 5 * DAY)
        fake, history = ready(jobs, now=NOW - 5 * DAY)
        checker = FastEqualsExact()
        checker.check(history, 7)
        # time passes: running jobs finish, new ones arrive, one changes state
        for row in jobs:
            if row["End"] is None:
                row["End"], row["State"] = NOW - 4 * DAY, "COMPLETED"
        jobs.extend(invented_jobs(8, count=80, now=NOW))
        fake.now = NOW
        fake.live = [j["JobIDRaw"] for j in jobs if j["End"] is None]
        history.refresh()
        history.build_rollup()
        self.assertIsNone(history._store.get_meta("dirty"))
        checker.check(history, 8)
        # measured usage written later changes the aggregates of its hours
        history.load_efficiency(Window(NOW - 20 * DAY, None))
        history.build_rollup()
        checker.check(history, 9)

    def test_dirty_hours_are_read_exactly(self):
        fake, history = ready(invented_jobs(10))
        store = history._store
        store.set_meta("dirty", [NOW - 10 * DAY, NOW - 9 * DAY])  # not rebuilt yet
        FastEqualsExact().check(history, 10)


class Layout(unittest.TestCase):
    def test_cache_from_before_the_aggregates_is_rebuilt(self):
        fake, history = ready(invented_jobs(11))
        store = history._store
        expected = build_report(store, history._as_of(), Window(None, None), exact=True,
                                gpu_tracked=True)
        # as left by an older version: no aggregates, rollup up to date
        store._write(lambda db: db.execute("DELETE FROM agg_hour"))
        store._write(lambda db: db.execute("DELETE FROM agg_start"))
        store._write(lambda db: db.execute("DELETE FROM meta WHERE k='agg'"))
        before = history.query(Window(None, None), group_by=())  # exact meanwhile
        self.assertEqual(before.total.gpu_seconds, expected.total.gpu_seconds)
        self.assertGreater(history.build_rollup(), 0)
        self.assertEqual(store.get_meta("agg"), AGG_VERSION)
        after = history.query(Window(None, None), group_by=())
        self.assertEqual(after.total.gpu_seconds, expected.total.gpu_seconds)
        self.assertEqual(after.total.attempts, expected.total.attempts)

    def test_old_layout_tables_are_replaced(self):
        fake, history = ready(invented_jobs(12))
        store = history._store
        store._write(lambda db: db.execute("DROP TABLE agg_hour"))
        store._write(lambda db: db.execute("CREATE TABLE agg_hour(hour INTEGER, dim INTEGER)"))
        store.set_meta("agg", AGG_VERSION - 1)
        history.build_rollup()
        FastEqualsExact().check(history, 12)


class Events(unittest.TestCase):
    def test_sql_baselines_equal_the_full_scan(self):
        for seed in (13, 14):
            fake, history = ready(invented_jobs(seed, count=600))
            store, as_of = history._store, history._as_of()
            for win in windows(seed)[:10]:
                for where in (Filter(), Filter(user="alice"), Filter(outcome=["failed"]),
                              Filter(account="chem", qos="normal")):
                    fast = events_module.find_events(store, as_of, win, where, 0)
                    slow = events_module.find_events_scan(store, as_of, win, where, 0)
                    self.assertEqual([(e.kind, e.time, e.end, e.sentence, e.values, e.job_id)
                                      for e in fast],
                                     [(e.kind, e.time, e.end, e.sentence, e.values, e.job_id)
                                      for e in slow], (win, where))


class JobLists(unittest.TestCase):
    def test_ranked_by_sqlite_equals_the_full_pass(self):
        from slurm_usage.history.listing import list_jobs
        for seed in (15, 16):
            fake, history = ready(invented_jobs(seed, count=500))
            store, as_of = history._store, history._as_of()
            for win in windows(seed, as_of)[:10]:
                for where in (Filter(), Filter(user="carol"), Filter(outcome=["completed"])):
                    for sort, by in (("gpu", "job"), ("cpu", "job"), ("start", "job"),
                                     ("start", "attempt"), ("gpu", "attempt")):
                        for descending in (True, False):
                            for limit, offset in ((20, 0), (15, 10), (5000, 0)):
                                fast = list_jobs(store, as_of, win, where, by, sort,
                                                 descending, limit, offset)
                                store.top_finished = None  # the full pass
                                try:
                                    slow = list_jobs(store, as_of, win, where, by, sort,
                                                     descending, limit, offset)
                                finally:
                                    del store.top_finished
                                self.assertEqual(fast.total, slow.total, (win, where, sort))
                                self.assertEqual(fast.rows, slow.rows, (win, where, sort))


class RunTypes(unittest.TestCase):
    def test_from_aggregates_equal_the_full_pass(self):
        from slurm_usage.history.runtypes import run_types
        for seed in (17, 18):
            fake, history = ready(invented_jobs(seed, count=500))
            history.load_efficiency(Window(NOW - 15 * DAY, None))
            history.build_rollup()
            store, as_of = history._store, history._as_of()
            for win in windows(seed, as_of)[:10]:
                for where in (Filter(), Filter(user="dave"), Filter(outcome=["completed"])):
                    for by in ("gpu_count", "account", "qos", "partition", "outcome"):
                        fast = run_types(store, as_of, win, where, by, gpu_tracked=True)
                        slow = run_types(store, as_of, win, where, by, gpu_tracked=True,
                                         exact=True)
                        label = (win, where, by)
                        self.assertEqual([t.key for t in fast], [t.key for t in slow], label)
                        for a, b in zip(fast, slow):
                            self.assertEqual((a.outcomes, a.median_runtime, a.median_wait,
                                              a.example), (b.outcomes, b.median_runtime,
                                                           b.median_wait, b.example), label)
                            self.assertAlmostEqual(a.share or 0, b.share or 0, places=9)
                            for name in FIELDS:
                                x, y = getattr(a.stats, name), getattr(b.stats, name)
                                if isinstance(x, float) or isinstance(y, float):
                                    self.assertAlmostEqual(x, y, delta=1e-6 * max(1, abs(y)))
                                else:
                                    self.assertEqual(x, y, (label, a.key, name))
                            self.assertEqual(a.stats.jobs, b.stats.jobs)


if __name__ == "__main__":
    unittest.main()
