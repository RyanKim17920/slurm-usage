"""Rollup ranges, run types and events."""

import unittest

from tests.test_history_support import H, T0, FakeSlurm, job, make_history

from slurm_usage.history import Filter, Window, name_family

DAY = 86400


class RollupTest(unittest.TestCase):
    def setUp(self):
        self.now = T0 + 20 * DAY + 30 * 60
        jobs = [
            job(1, start=self.now - 2 * H - 1800, end=self.now - 1800, gpus=2, account="x",
                user="a", uid=1),
            job(2, start=self.now - 8 * DAY, end=self.now - 8 * DAY + 3 * H, gpus=1,
                account="y", user="b", uid=2),
            job(3, start=self.now - 3 * DAY, end=None, state="RUNNING", gpus=4, account="x",
                user="c", uid=3),
        ]
        self.fake = FakeSlurm(jobs, now=self.now, live=[3])
        self.h = make_history(self.fake)
        self.h.ROLLUP_BETWEEN = False  # these tests build the rollup themselves
        self.h.refresh()

    def ranges(self):
        return {r.name: r for r in self.h.overview(gpu_capacity=16).ranges}

    def test_ranges_match_direct_queries(self):
        self.h.build_rollup()
        ranges = self.ranges()
        for name, summary in ranges.items():
            direct = self.h.query(Window(summary.start, summary.end), group_by=["account"])
            self.assertEqual(summary.total.gpu_seconds, direct.total.gpu_seconds, name)
            for (account,), stats in direct.group("account").items():
                if stats.gpu_seconds:
                    self.assertEqual(summary.accounts[account].amounts.gpu_seconds,
                                     stats.gpu_seconds)
            self.assertTrue(summary.complete)
        week = ranges["week"]
        self.assertEqual(week.end - week.start, 7 * DAY)
        self.assertIsNotNone(week.previous)
        self.assertIsNone(ranges["all"].previous)
        self.assertIn("today", week.label)
        self.assertEqual(ranges["week"].accounts["x"].running_gpus, 4)
        self.assertEqual(set(ranges["day"].accounts["x"].users), {"a", "c"})
        self.assertAlmostEqual(week.gpu_utilization, week.total.gpu_seconds / (16 * 7 * DAY))

    def check_series(self, history):
        history.build_rollup()
        ranges = {r.name: r for r in history.overview().ranges}
        expected = {"day": (H, 24), "week": (H, 168), "month": (DAY, 30), "year": (DAY, 365)}
        for name, summary in ranges.items():
            step, count = expected.get(name, (DAY, -(-(summary.end - summary.start) // DAY)))
            self.assertEqual((summary.series_step, len(summary.gpu_series)), (step, count), name)
            self.assertEqual(sum(summary.gpu_series), summary.total.gpu_seconds, name)
            self.assertEqual(sum(summary.cpu_series), summary.total.cpu_seconds, name)
            self.assertEqual(summary.spark(), summary.gpu_series)
            # the newest bin is exactly the last step before the end
            last = history.query(Window(summary.end - step, summary.end))
            self.assertEqual(summary.gpu_series[-1], last.total.gpu_seconds, name)
        self.assertEqual(ranges["year"].gpu_series[-30:], ranges["month"].gpu_series)
        self.assertEqual(ranges["week"].gpu_series[-24:], ranges["day"].gpu_series)

    def test_series_per_range_from_the_rollup(self):
        self.check_series(self.h)

    def test_series_from_the_sqlite_rollup(self):
        from tests.test_history_support import temp_dir
        history = make_history(self.fake, persist=True, cache_dir=temp_dir(self))
        history.refresh()
        self.check_series(history)

    def test_one_utilization_denominator_for_every_range(self):
        self.h.refresh(None)  # all of history: the floor is the first record
        self.h.build_rollup()
        ranges = self.ranges()
        year, everything = ranges["year"], ranges["all"]
        # all usage lies within the year, so both describe the same time and usage
        self.assertEqual(year.total.gpu_seconds, everything.total.gpu_seconds)
        self.assertAlmostEqual(year.gpu_utilization, everything.gpu_utilization)
        direct = {name: self.h.query(Window(r.start if name != "all" else None, r.end))
                  .utilization(16).fraction for name, r in ranges.items()}
        self.assertAlmostEqual(direct["year"], direct["all"])
        self.assertAlmostEqual(direct["year"], year.gpu_utilization)

    def test_incremental_rebuild_only_where_changed(self):
        self.assertGreater(self.h.build_rollup(), 0)
        self.assertEqual(self.h.build_rollup(), 0)
        self.fake.now += 2 * H
        self.h._clock = lambda: self.fake.now
        self.h.refresh()
        rebuilt = self.h.build_rollup()
        self.assertGreater(rebuilt, 0)
        self.assertLess(rebuilt, 24 * 4)
        self.test_ranges_match_direct_queries()

    def test_unbuilt_rollup_is_incomplete(self):
        self.assertFalse(self.ranges()["week"].complete)


class HugeMemoryTest(unittest.TestCase):
    def test_byte_seconds_beyond_64_bits_stay_exact(self):
        now = T0 + 800 * DAY
        fake = FakeSlurm([job(1, start=T0, end=now - DAY, mem="1000T", gpus=0, cpus=1)],
                         now=now, live=[])
        h = make_history(fake)
        h.refresh()
        h.build_rollup()
        everything = [r for r in h.overview().ranges if r.name == "all"][0]
        direct = h.query(Window(everything.start, everything.end)).total.mem_byte_seconds
        self.assertGreater(direct, 2 ** 63)
        self.assertEqual(everything.total.mem_byte_seconds, direct)


class RunTypeTest(unittest.TestCase):
    def test_name_family(self):
        self.assertEqual(name_family("train_seed3_2026-01-02"), "train_seed<n>_<date>")
        self.assertEqual(name_family("eval-1f3a9c7e"), "eval-<hex>")
        self.assertEqual(name_family("eval-deadbeefcafe9"), "eval-<hex>")
        self.assertEqual(name_family("x_1_2_3"), "x_<n>")
        self.assertEqual(name_family(""), "(no name)")

    def test_types_sum_and_drill(self):
        jobs = [job(1, start=T0, end=T0 + H, gpus=8, name="pretrain_7", eligible=T0 - H),
                job(2, start=T0, end=T0 + 3 * H, gpus=8, name="pretrain_8", state="FAILED"),
                job(3, start=T0, end=T0 + H, gpus=1, name="eval_a1b2c3d4")]
        fake = FakeSlurm(jobs, live=[])
        h = make_history(fake)
        h.refresh()
        w = Window(T0, T0 + DAY)
        for by in ("gpu_count", "name_family", "account", "qos", "partition", "outcome",
                   "node_count"):
            types = h.run_types(w, by=by)
            self.assertEqual(sum(t.stats.gpu_seconds for t in types),
                             h.query(w).total.gpu_seconds, by)
        types = {t.key: t for t in h.run_types(w, by="name_family")}
        pretrain = types["pretrain_<n>"]
        self.assertEqual((pretrain.stats.attempts, pretrain.stats.gpu_seconds), (2, 32 * H))
        self.assertAlmostEqual(pretrain.failure_rate, 0.5)
        self.assertAlmostEqual(pretrain.share, 32 / 33.0)
        self.assertEqual(pretrain.median_runtime, H)
        self.assertEqual(pretrain.median_wait, 0)
        page = h.jobs(w, run_type=("name_family", "pretrain_<n>"))
        self.assertEqual(sorted(r.job_id for r in page.rows), ["1", "2"])
        with self.assertRaises(ValueError):
            h.run_types(w, by="color")


class EventTest(unittest.TestCase):
    def test_events_against_own_history(self):
        base = T0 + 10 * DAY
        jobs = [job(i, start=T0 + i * DAY, end=T0 + i * DAY + H, gpus=1, user="a", uid=1)
                for i in range(1, 9)]
        jobs += [
            job(50, start=base, end=base + 20 * H, gpus=8, user="a", uid=1, qos="high"),
            job(51, start=base + DAY, end=base + DAY + H, state="PREEMPTED", index=1,
                submit=base + DAY, user="a", uid=1),
            job(51, start=base + DAY + 2 * H, end=base + DAY + 3 * H, state="PREEMPTED",
                index=2, submit=base + DAY + H, user="a", uid=1),
            job(51, start=base + DAY + 4 * H, end=base + DAY + 5 * H, index=3,
                submit=base + DAY + 3 * H, user="a", uid=1),
        ]
        jobs += [job(60 + i, start=base + 2 * DAY + i * 60, end=base + 2 * DAY + i * 60 + 30,
                     state="FAILED", ExitCode="2:0", user="a", uid=1) for i in range(6)]
        fake = FakeSlurm(jobs, now=base + 9 * DAY, live=[])
        h = make_history(fake)
        h.refresh()
        events = h.events(Window(base, base + 7 * DAY), Filter(user="a"))
        kinds = {e.kind for e in events}
        for kind in ("largest_allocation", "longest_run", "preemption_run", "failure_burst",
                     "idle_gap", "first_qos", "record_day", "change"):
            self.assertIn(kind, kinds)
        self.assertEqual([e.time for e in events], sorted(e.time for e in events))
        largest = [e for e in events if e.kind == "largest_allocation"][0]
        self.assertTrue(largest.values["record"])
        burst = [e for e in events if e.kind == "failure_burst"][0]
        self.assertEqual((burst.values["count"], burst.values["exit_code"]), (6, "2:0"))
        self.assertEqual(h.query(burst.window, burst.where).total.attempts, 6)
        self.assertTrue(all(e.sentence and "\x1b" not in e.sentence for e in events))

    def test_overlap_before_the_window_is_not_a_baseline(self):
        base = T0 + 10 * DAY
        jobs = [job(i, start=T0 + i * DAY, end=T0 + i * DAY + 4 * H, gpus=8, user="a", uid=1)
                for i in range(1, 10)]
        jobs += [job(50, start=base + H, end=base + 3 * H, gpus=1, user="a", uid=1,
                     qos="high")]
        fake = FakeSlurm(jobs, now=base + 7 * DAY, live=[])
        h = make_history(fake)
        # the loader's window starts a little before the one asked about later (its clock
        # was earlier, and ingest reads an overlap): a sliver of history, not a baseline
        h.refresh(Window(base - 2 * H, base + 7 * DAY))
        covered = h.info().covered_from
        self.assertLess(covered, base)
        self.assertGreater(covered, base - DAY)
        events = h.events(Window(base, base + 7 * DAY), Filter(user="a"))
        kinds = {e.kind for e in events}
        for kind in ("record_day", "change", "first_qos", "first_account"):
            self.assertNotIn(kind, kinds)
        self.assertFalse(any(e.values.get("record") for e in events))
        self.assertIn("no_baseline", kinds)
        # once the previous equal period is loaded, it is a baseline again
        h.refresh(Window(base - 7 * DAY, base + 7 * DAY))
        kinds = {e.kind for e in h.events(Window(base, base + 7 * DAY), Filter(user="a"))}
        self.assertIn("change", kinds)
        self.assertNotIn("no_baseline", kinds)


if __name__ == "__main__":
    unittest.main()
