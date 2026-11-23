"""The quicker paths behind the drill screens answer exactly what the plain ones do:
run-type and job-name-family lists narrowed in SQLite, the newest attempts read newest
first, searches prefiltered by SQLite, run types by family from the aggregates, medians
worked out on first use, a selected job's detail fetched ahead. Invented data."""

import pickle
import time
import unittest

from tests.test_perf_aggregates import FIELDS, ready, invented_jobs, windows

from slurm_usage.history import Filter, Window
from slurm_usage.history.listing import list_jobs
from slurm_usage.history.runtypes import run_types


def named(seed: int, count: int = 500):
    """invented_jobs with some names a family or a search has to get right: empty, only
    spaces, upper case and characters SQLite does not lower."""
    jobs = invented_jobs(seed, count=count)
    odd = ["", "   ", "TRAIN_7", "Étude_3", "Kelvin_run", "eval"]
    for at, found in enumerate(jobs):
        if at % 7 == 0:
            found["JobName"] = odd[(at // 7) % len(odd)]
    return jobs


class _Plain:
    """The store as list_jobs used it before: no SQL narrowing, no SQL ranking."""

    def __init__(self, store):
        self._store = store

    def __getattr__(self, name):
        if name in ("top_finished", "has_families"):
            raise AttributeError(name)
        return getattr(self._store, name)

    def scan(self, low, high, dim_ids=None, usage=False, ordered=True, by_start=False,
             extra=False, **narrowing):
        return self._store.scan(low, high, dim_ids, usage=usage,
                                ordered=bool(ordered) and ordered is True,
                                by_start=by_start, extra=extra)


class NarrowedLists(unittest.TestCase):
    def test_run_types_and_searches_equal_the_plain_pass(self):
        for seed in (21, 22):
            fake, history = ready(named(seed))
            store, as_of = history._store, history._as_of()
            plain = _Plain(store)
            kinds = [None, ("gpu_count", "2"), ("gpu_count", "0"), ("account", "chem"),
                     ("qos", "high"), ("outcome", "completed"), ("outcome", "running"),
                     ("outcome", "preempted"), ("name_family", "train_<n>"),
                     ("name_family", "eval"), ("name_family", "(no name)"),
                     ("name_family", "TRAIN_<n>"), ("name_family", "nothing like it")]
            for win in windows(seed, as_of)[:7]:
                for where in (Filter(), Filter(user="carol"), Filter(outcome=["completed"])):
                    for run_type in kinds:
                        for match in ("", "tra", "1", "ali", "étude", "kelvin"):
                            for sort, by in (("gpu", "job"), ("start", "attempt"),
                                             ("start", "job")):
                                label = (win, where, run_type, match, sort, by)
                                fast = list_jobs(store, as_of, win, where, by, sort, True, 30,
                                                 0, run_type=run_type, match=match)
                                slow = list_jobs(plain, as_of, win, where, by, sort, True, 30,
                                                 0, run_type=run_type, match=match)
                                self.assertEqual(fast.total, slow.total, label)
                                self.assertEqual(fast.rows, slow.rows, label)

    def test_newest_attempts_uncounted_equal_counted(self):
        fake, history = ready(named(23))
        store, as_of = history._store, history._as_of()
        for walk in (False, True):  # sorting the selection, or walking the start index
            store.NEWEST_PAGE = 0 if walk else 10 ** 9
            try:
                for win in windows(23, as_of)[:8]:
                    for where in (Filter(), Filter(user="alice")):
                        for run_type in (None, ("gpu_count", "4"), ("name_family", "eval"),
                                         ("outcome", "completed")):
                            for limit, offset in ((10, 0), (25, 5), (5000, 0)):
                                counted = list_jobs(store, as_of, win, where, "attempt",
                                                    "start", True, limit, offset,
                                                    run_type=run_type)
                                quick = list_jobs(store, as_of, win, where, "attempt",
                                                  "start", True, limit, offset,
                                                  run_type=run_type, count=False)
                                self.assertIsNone(quick.total)
                                self.assertEqual(quick.rows, counted.rows,
                                                 (walk, win, where, run_type, limit))
            finally:
                del store.NEWEST_PAGE


class FamilyTypes(unittest.TestCase):
    def test_families_from_the_aggregates_equal_the_full_pass(self):
        for seed in (24, 25):
            fake, history = ready(named(seed))
            store, as_of = history._store, history._as_of()
            for win in windows(seed, as_of)[:9]:
                for where in (Filter(), Filter(user="dave"), Filter(outcome=["completed"])):
                    fast = run_types(store, as_of, win, where, "name_family", gpu_tracked=True)
                    slow = run_types(store, as_of, win, where, "name_family", gpu_tracked=True,
                                     exact=True)
                    label = (win, where)
                    self.assertEqual([t.key for t in fast], [t.key for t in slow], label)
                    for a, b in zip(fast, slow):
                        self.assertEqual((a.outcomes, a.median_runtime, a.median_wait),
                                         (b.outcomes, b.median_runtime, b.median_wait), label)
                        self.assertAlmostEqual(a.share or 0, b.share or 0, places=9)
                        for name in FIELDS:
                            x, y = getattr(a.stats, name), getattr(b.stats, name)
                            if isinstance(x, float) or isinstance(y, float):
                                self.assertAlmostEqual(x, y, delta=1e-6 * max(1.0, abs(y)))
                            else:
                                self.assertEqual(x, y, (label, a.key, name))

    def test_family_aggregates_follow_new_rows(self):
        jobs = named(26, count=300)
        for found in jobs[200:260]:  # names never seen before the refresh
            found["JobName"] = "fresh_%s" % found["JobIDRaw"]
        fake, history = ready(jobs[:200])
        fake.jobs[:] = jobs
        history.refresh()
        history.build_rollup()
        store, as_of = history._store, history._as_of()
        for win in (Window(None, None), Window(as_of - 9 * 86400, None)):
            fast = [(t.key, t.stats.gpu_seconds, t.stats.attempts)
                    for t in run_types(store, as_of, win, None, "name_family")]
            slow = [(t.key, t.stats.gpu_seconds, t.stats.attempts)
                    for t in run_types(store, as_of, win, None, "name_family", exact=True)]
            self.assertEqual(fast, slow)
            for run_type in (("name_family", "fresh_<n>"), ("name_family", "eval")):
                quick = list_jobs(store, as_of, win, None, "job", "gpu", True, 1000,
                                  run_type=run_type)
                plain = list_jobs(_Plain(store), as_of, win, None, "job", "gpu", True, 1000,
                                  run_type=run_type)
                self.assertEqual((quick.total, quick.rows), (plain.total, plain.rows))


    def test_a_cache_without_them_builds_them_once_from_its_rows(self):
        from slurm_usage.history.runtypes import _fast_types
        from slurm_usage.history.store import FAMILY_VERSION
        fake, history = ready(named(29, count=200))
        store, as_of = history._store, history._as_of()
        store.set_meta("agg_fam", None)  # as a cache from before the family aggregates
        win, where = Window(None, None), Filter()
        self.assertIsNone(_fast_types(store, as_of, win, where, "name_family", "gpu", True,
                                      lambda: False, None))
        calls = len(fake.calls)
        history.build_rollup()
        self.assertEqual(len(fake.calls), calls)  # from the cache, not from Slurm
        self.assertEqual(store.get_meta("agg_fam"), FAMILY_VERSION)
        fast = _fast_types(store, as_of, win, where, "name_family", "gpu", True,
                           lambda: False, None)
        slow = run_types(store, as_of, win, where, "name_family", exact=True)
        self.assertIsNotNone(fast)
        self.assertEqual([(t.key, t.stats.gpu_seconds, t.stats.attempts) for t in fast],
                         [(t.key, t.stats.gpu_seconds, t.stats.attempts) for t in slow])


class LazyMedians(unittest.TestCase):
    def test_worked_out_once_on_first_read(self):
        fake, history = ready(invented_jobs(27, count=300))
        store, as_of = history._store, history._as_of()
        win = Window(None, None)
        calls = []
        real = store.type_medians

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        store.type_medians = counted
        try:
            found = run_types(store, as_of, win, None, "gpu_count")
            self.assertEqual(calls, [])  # nothing read yet
            found[1].example = "kept"  # set before: stays
            first = found[0].median_runtime
            self.assertEqual(len(calls), 1)
            self.assertEqual([t.median_wait for t in found] and len(calls), 1)
            self.assertEqual(found[1].example, "kept")
        finally:
            del store.type_medians
        exact = run_types(store, as_of, win, None, "gpu_count", exact=True)
        self.assertEqual(first, exact[0].median_runtime)
        again = pickle.loads(pickle.dumps(found[0]))
        self.assertEqual((again.key, again.median_runtime, again.median_wait, again.example),
                         (found[0].key, found[0].median_runtime, found[0].median_wait,
                          found[0].example))

    def test_screens_leave_them_out(self):
        from slurm_usage.ui import data as hd
        fake, history = ready(invented_jobs(28, count=200))
        data = hd.HistoryData(history)
        calls = []
        real = history._store.type_medians
        history._store.type_medians = lambda *a, **k: calls.append(1) or real(*a, **k)
        try:
            data.run_types("all", None, "gpu_count", medians=False)
            self.assertEqual(calls, [])
            with_medians = data.run_types("all", None, "gpu_count")
            self.assertEqual(len(calls), 1)
            self.assertTrue(any(t.runtime is not None for t in with_medians))
        finally:
            del history._store.type_medians


class AheadOfEnter(unittest.TestCase):
    def test_a_selected_job_is_fetched_before_it_is_opened(self):
        from tests.ui_fakes import FakeCollector, real_history
        from slurm_usage.ui import data as hd
        from slurm_usage.ui.app import App
        from slurm_usage.ui.canvas import Canvas
        history, fake = real_history(days=10)
        data = hd.HistoryData(history)
        data.load("week")
        app = App(FakeCollector(), data, "week", start="queue", history="full")
        app.PREFETCH_AFTER = 0.0
        asked = []
        data.job_detail = lambda job_id: asked.append(job_id) or None
        try:
            for _ in range(200):
                app.poll()
                app.render(Canvas(160, 50))
                if asked:
                    break
                time.sleep(0.01)
            self.assertTrue(asked)
            job_id = asked[0]
            self.assertIsNotNone(app.loader.peek(("o", "detail", job_id)))
            before = len(asked)
            app.push("job", {"job_id": job_id}, "job")
            app.render(Canvas(160, 50))
            self.assertEqual(len(asked), before)  # the job screen reuses it
        finally:
            app.stop()

    def test_long_lists_are_laid_out_once_per_answer(self):
        from tests.ui_fakes import FakeCollector, real_history
        from slurm_usage.ui import data as hd
        from slurm_usage.ui import screens
        from slurm_usage.ui.app import App
        from slurm_usage.ui.canvas import Canvas
        from slurm_usage.ui.loader import Loader
        history, fake = real_history(days=10)
        data = hd.HistoryData(history)
        data.load("week")
        user = history.jobs(Window(None, None), None, limit=1).rows[0].user
        app = App(FakeCollector(), data, "week", start="person", user=user,
                  background=False)
        app.loader = Loader(workers=0)
        app.view.extra["dim"] = "name_family"
        built = []
        real = screens.breakdown_lines
        screens.breakdown_lines = lambda *a, **k: built.append(1) or real(*a, **k)
        try:
            for _ in range(5):
                app.render(Canvas(160, 50))
        finally:
            screens.breakdown_lines = real
        self.assertEqual(len(built), 1)
        app.color = not app.color  # another layout: laid out again
        screens.breakdown_lines = lambda *a, **k: built.append(1) or real(*a, **k)
        try:
            app.render(Canvas(160, 50))
        finally:
            screens.breakdown_lines = real
        self.assertEqual(len(built), 2)

    def test_the_loader_counts_answers_so_the_last_one_is_drawn(self):
        from slurm_usage.ui.loader import Loader
        loader = Loader(workers=0)
        before = loader.finished
        loader.get(("h", "x"), lambda: 1)
        loader.get(("h", "y"), lambda: 1 / 0)  # a failure is something to draw too
        self.assertEqual(loader.finished, before + 2)

    def test_switching_history_mode_keeps_what_is_shown(self):
        from tests.ui_fakes import FakeCollector, real_history
        from slurm_usage.ui import data as hd
        from slurm_usage.ui.app import App
        history, fake = real_history(days=10)
        data = hd.HistoryData(history)
        app = App(FakeCollector(), data, "week", history="full", background=False)
        generation = app.loader.generation
        for mode in ("recent", "off", "full"):
            app.set_history(mode)
        self.assertEqual(app.loader.generation, generation)


if __name__ == "__main__":
    unittest.main()
