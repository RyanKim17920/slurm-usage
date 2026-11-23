"""Answers kept across runs, refreshes that change nothing, and the cheaper paths behind
views: each gives exactly what the plain computation gives. Invented data only."""

import os
import pickle
import unittest

from tests.test_history_support import FakeSlurm, make_history, temp_dir
from tests.test_perf_aggregates import NOW, invented_jobs, ready, windows

from slurm_usage.history import Filter, Window
from slurm_usage.history import environment
from slurm_usage.history.aggregate import LazyStats, build_report, kind_bins
from slurm_usage.history.rollup import with_capacity
from slurm_usage.history.stats import Stats
from slurm_usage.ui import data as hd
from slurm_usage.ui.loader import Loader

DAY = 86400


class Persisted(object):
    """History opened on a cache directory, the way the app does."""

    @staticmethod
    def history(case, jobs):
        fake = FakeSlurm(jobs, now=NOW, live=[j["JobIDRaw"] for j in jobs if j["End"] is None])
        directory = temp_dir(case)
        history = make_history(fake, persist=True, cache_dir=directory)
        history.refresh()
        history.build_rollup()
        return fake, history, directory


class ResultCacheTest(unittest.TestCase):
    def test_answers_come_back_for_the_same_cache_only(self):
        fake, history, directory = Persisted.history(self, invented_jobs(30, count=120))
        cache = hd.ResultCache(history)
        report = history.query(Window(NOW - 7 * DAY, None), group_by=(("user",),))
        key = ("h", "report", "week", ())
        cache.save(key, report)
        self.assertIsNone(hd.ResultCache(history).load(key))  # not written yet
        cache.flush()
        again = hd.ResultCache(history).load(key)
        self.assertEqual(again.total.gpu_seconds, report.total.gpu_seconds)
        self.assertEqual(set(again.groups[("user",)]), set(report.groups[("user",)]))
        self.assertTrue(oct(os.stat(os.path.join(directory, "views-v1.sqlite")).st_mode)
                        .endswith("600"))
        # another cache file's answers are never shown
        other = hd.ResultCache(history)
        other._ident = lambda: "another cache"
        self.assertIsNone(other.load(key))
        # nor while inactive (anonymized), nor for keys that are not history answers
        self.assertIsNone(hd.ResultCache(history, lambda: False).load(key))
        cache.save(("l", "snap"), report)
        self.assertIsNone(cache.load(("l", "snap")))

    def test_without_a_cache_file_nothing_is_kept(self):
        _, history = ready(invented_jobs(31, count=50))  # in memory
        cache = hd.ResultCache(history)
        cache.save(("h", "x"), 1)
        cache.flush()
        self.assertIsNone(cache.load(("h", "x")))

    def test_lazy_fields_pickle_as_plain_stats(self):
        _, history = ready(invented_jobs(32, count=200))
        report = history.query(Window(None, None))
        self.assertIsInstance(report.total, LazyStats)
        copy = pickle.loads(pickle.dumps(report))
        self.assertIs(type(copy.total), Stats)
        self.assertEqual(copy.total.gpu_seconds, report.total.gpu_seconds)
        self.assertEqual(list(copy.waits), list(report.waits))

    def test_loader_shows_the_kept_answer_and_still_recomputes(self):
        kept = {("h", "a"): "old"}

        class Seed:
            saved = {}

            def load(self, key):
                return kept.get(key)

            def save(self, key, value):
                self.saved[key] = value
        seed = Seed()
        loader = Loader(workers=0, seed=seed)
        calls = []
        entry = loader.get(("h", "a"), lambda: calls.append(1) or "new")
        self.assertEqual(entry.value, "new")  # inline: recomputed at once
        self.assertEqual(calls, [1])
        self.assertEqual(seed.saved[("h", "a")], "new")


class PriorityTest(unittest.TestCase):
    def test_work_for_a_screen_left_behind_is_dropped(self):
        import threading
        gate = threading.Event()
        loader = Loader(workers=1)
        self.addCleanup(loader.close)
        loader.get(("h", "busy"), gate.wait)  # occupies the only worker
        ran = []
        waiting = loader.get(("h", "old"), lambda: ran.append("old"))
        self.assertTrue(waiting.busy)
        self.assertEqual(loader.drop_queued(), 1)
        self.assertFalse(waiting.busy)
        again = loader.get(("h", "old"), lambda: ran.append("again") or 1)  # asked again
        self.assertTrue(again.busy)
        gate.set()
        again.future.result(timeout=5)
        self.assertEqual(ran, ["again"])


class UnchangedTest(unittest.TestCase):
    def test_writing_the_same_meta_changes_nothing(self):
        _, history = ready(invented_jobs(33, count=50))
        store = history._store
        store.set_meta("probe", [1, 2])
        version, generation = store.version, store.generation()
        store.set_meta("probe", [1, 2])
        self.assertEqual((store.version, store.generation()), (version, generation))
        store.set_meta("lookback_at", 12345)  # bookkeeping: no answer changes
        self.assertEqual(store.version, version)
        store.set_meta("probe", [3])
        self.assertGreater(store.generation(), generation)

    def test_a_refresh_that_brings_nothing_does_not_recompute_views(self):
        fake, history = ready(invented_jobs(34, count=80))
        data = hd.HistoryData(history, clock=lambda: fake.now)
        data.load("week")
        generation = data.generation
        data.load("month", update=False)  # already covered: nothing changes
        self.assertEqual(data.generation, generation)

    def test_overview_for_another_capacity_is_the_same_as_computing_it(self):
        _, history = ready(invented_jobs(35, count=200))
        for gpus, cpus in ((16, 64), (None, 64), (8, None)):
            direct = history.overview(gpus, cpus)
            derived = with_capacity(history.overview(), gpus, cpus)
            for a, b in zip(direct.ranges, derived.ranges):
                self.assertEqual((a.gpu_utilization, a.cpu_utilization,
                                  a.previous_gpu_utilization),
                                 (b.gpu_utilization, b.cpu_utilization,
                                  b.previous_gpu_utilization))


class KindSeriesTest(unittest.TestCase):
    def test_binned_kinds_equal_the_full_pass(self):
        _, history = ready(invented_jobs(36))
        store, as_of = history._store, history._as_of()
        for win in windows(36, as_of):
            for where in (Filter(), Filter(user="bob"), Filter(outcome=["completed"])):
                for unit in ("hour", "3hour", "day", "week"):
                    make = lambda start, end, u=unit: hd.bin_edges(start, end, u)  # noqa: E731
                    edges, fast = kind_bins(store, as_of, win, where, make, True)
                    store.has_agg = False  # the full pass
                    try:
                        same, slow = kind_bins(store, as_of, win, where, make, True)
                    finally:
                        del store.has_agg
                    self.assertEqual(edges, same)
                    self.assertEqual({k: v for k, v in fast.items() if any(v)},
                                     {k: v for k, v in slow.items() if any(v)}, (win, where))
                    # and they add up to the window's GPU-seconds
                    report = build_report(store, as_of, win, where, (), gpu_tracked=True)
                    self.assertEqual(sum(sum(v) for v in fast.values()),
                                     sum(report.values["gpu"]))

    def test_kinds_by_role_and_size(self):
        _, history = ready(invented_jobs(37))
        edges, found = history.kind_series(Window(NOW - 30 * DAY, None), None,
                                           kinds=("gpu_count", "account", "outcome"))
        report = history.query(Window(NOW - 30 * DAY, None), group_by=(("account",),))
        for kind in ("gpu_count", "account", "outcome"):
            self.assertEqual(sum(sum(v) for v in found[kind].values()),
                             report.total.gpu_seconds, kind)
        for (account,), stats in report.group("account").items():
            self.assertEqual(sum(found["account"].get(account, [0])), stats.gpu_seconds)


class ProbeCacheTest(unittest.TestCase):
    def test_a_probe_is_reused_for_a_while(self):
        directory = temp_dir(self)
        fake = FakeSlurm([], now=NOW)
        clock = [1000.0]
        first = environment.cached_probe(fake, directory, lambda: clock[0])
        asked = len(fake.calls)
        again = environment.cached_probe(fake, directory, lambda: clock[0] + 10)
        self.assertEqual(len(fake.calls), asked)  # nothing asked of Slurm
        self.assertEqual((again.cluster, again.visibility, again.identity, again.fields),
                         (first.cluster, first.visibility, first.identity, first.fields))
        self.assertEqual(again.invoker, first.invoker)
        environment.cached_probe(fake, directory, lambda: clock[0] + environment.PROBE_TTL + 1)
        self.assertGreater(len(fake.calls), asked)  # stale: asked again
        self.assertTrue(oct(os.stat(os.path.join(directory, "slurm-usage", "probe-v1.json")
                                    if os.path.exists(os.path.join(directory, "slurm-usage"))
                                    else os.path.join(directory, "probe-v1.json")).st_mode)
                        .endswith("600"))


if __name__ == "__main__":
    unittest.main()
