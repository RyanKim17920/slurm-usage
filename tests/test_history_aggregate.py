"""Accounting arithmetic: clipping, attempts, groupings, buckets, DST, outcomes."""

import unittest

from tests.test_history_support import H, T0, FakeSlurm, job, make_history, step, timezone

from slurm_usage.history import Filter, HistoryError, NONE, OUTCOMES, Window

DAY = 86400


def loaded(jobs, live=(), now=None, **kw):
    fake = FakeSlurm(jobs, now=now or T0 + 10 * DAY, live=None if live is None else list(live), **kw)
    history = make_history(fake)
    history.refresh()
    return fake, history


def assert_sums(case, report):
    total = report.total
    for spec, groups in report.groups.items():
        if "node" in spec:
            continue
        for name in ("gpu_seconds", "cpu_seconds", "mem_byte_seconds", "attempts", "seconds"):
            case.assertEqual(sum(getattr(s, name) for s in groups.values()),
                             getattr(total, name), (spec, name))
    for metric in ("gpu", "cpu", "mem"):
        if total.amount(metric) is None:
            continue
        for unit in ("hour", "day"):
            case.assertEqual(sum(b.value for b in report.buckets(metric, unit)),
                             total.amount(metric))
        case.assertEqual(sum(map(sum, report.heatmap(metric))), total.amount(metric))
        case.assertEqual(report.cumulative(metric)[-1][1], total.amount(metric))


class ClipTest(unittest.TestCase):
    def test_clipped_to_window(self):
        _, h = loaded([job(1, start=T0, end=T0 + 10 * H, gpus=2, cpus=8)])
        r = h.query(Window(T0 + 4 * H, T0 + 6 * H))
        self.assertEqual(r.total.gpu_seconds, 2 * 2 * H)
        self.assertEqual(r.total.cpu_seconds, 8 * 2 * H)
        self.assertEqual(r.total.attempts, 1)
        assert_sums(self, r)

    def test_job_spanning_the_window_from_both_sides(self):
        _, h = loaded([job(1, start=T0 - DAY, end=T0 + 3 * DAY, gpus=1)], now=T0 + 5 * DAY)
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual(r.total.gpu_seconds, DAY)
        self.assertEqual(r.total.active_seconds, DAY)

    def test_suspended_time_is_subtracted_and_splits_add_up(self):
        jobs = [job(1, start=T0, end=T0 + 10 * H, suspended=4 * H, gpus=1)]
        _, h = loaded(jobs)
        whole = h.query(Window(T0, T0 + DAY)).total.gpu_seconds
        self.assertEqual(whole, 6 * H)
        parts = sum(h.query(Window(T0 + i * 3 * H, T0 + (i + 1) * 3 * H)).total.gpu_seconds
                    for i in range(8))
        self.assertEqual(parts, whole)
        self.assertTrue(any(n.code == "suspension" for n in h.query().notes))

    def test_pending_has_no_usage_but_is_counted(self):
        _, h = loaded([job(1, state="PENDING", start=None, end=None, submit=T0)])
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual(r.total.gpu_seconds, 0)
        self.assertEqual(r.total.attempts, 0)
        self.assertEqual(r.total.pending, 1)

    def test_cancelled_before_start_counts_as_never_started(self):
        _, h = loaded([job(1, state="CANCELLED by 1", start=None, end=T0 + H, submit=T0)])
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual((r.total.attempts, r.total.never_started), (0, 1))

    def test_zero_length_allocation_is_an_attempt(self):
        _, h = loaded([job(1, start=T0 + H, end=T0 + H, state="FAILED")])
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual((r.total.attempts, r.total.gpu_seconds), (1, 0))


class RunningTest(unittest.TestCase):
    def test_running_counts_to_as_of(self):
        now = T0 + 5 * H
        _, h = loaded([job(1, start=T0, end=None, state="RUNNING", gpus=4)], live=[1], now=now)
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual(r.total.gpu_seconds, 4 * 5 * H)
        self.assertEqual(r.total.running, 1)
        self.assertEqual(r.end, now)
        self.assertEqual(r.outcomes["running"].gpu_seconds, 4 * 5 * H)

    def test_runaway_not_in_squeue_is_flagged_not_counted(self):
        now = T0 + 5 * H
        _, h = loaded([job(1, start=T0, end=None, state="RUNNING")], live=[], now=now)
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual(r.total.gpu_seconds, 0)
        self.assertEqual(r.total.unresolved, 1)
        self.assertTrue(any(n.code == "unresolved" for n in r.notes))
        rows = h.attempts("1")
        self.assertEqual(rows[0].problem, "runaway")

    def test_ended_state_without_end_is_not_counted(self):
        _, h = loaded([job(1, start=T0, end=None, state="COMPLETED")], live=[1])
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual((r.total.gpu_seconds, r.total.unresolved), (0, 1))

    def test_far_past_limit_without_squeue_is_stale(self):
        now = T0 + 5 * DAY
        _, h = loaded([job(1, start=T0, end=None, state="RUNNING", limit=60)], live=None,
                      now=now)
        r = h.query(Window(T0, now))
        self.assertEqual(r.total.unresolved, 1)
        self.assertEqual(h.attempts("1")[0].problem, "stale")

    def test_requeued_older_open_attempt_is_runaway(self):
        jobs = [job(1, start=T0, end=None, state="RUNNING", submit=T0, index=11),
                job(1, start=T0 + 2 * H, end=None, state="RUNNING", submit=T0 + H, index=12)]
        _, h = loaded(jobs, live=[1], now=T0 + 3 * H)
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual(r.total.gpu_seconds, H)
        self.assertEqual(r.total.unresolved, 1)


class AttemptsTest(unittest.TestCase):
    def jobs(self):
        return [
            job(1, start=T0, end=T0 + H, state="PREEMPTED", submit=T0, index=1, restarts=0),
            job(1, start=T0 + 2 * H, end=T0 + 3 * H, state="PREEMPTED", submit=T0 + H, index=2,
                restarts=1),
            job(1, start=T0 + 4 * H, end=T0 + 6 * H, state="COMPLETED", submit=T0 + 3 * H,
                index=3, restarts=2),
            job(2, start=T0, end=T0 + H, label="2_0"),
            job(3, start=T0, end=T0 + H, label="2_1"),
        ]

    def test_attempts_versus_distinct_jobs(self):
        _, h = loaded(self.jobs())
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual(r.total.attempts, 5)
        self.assertEqual(r.total.jobs, 3)
        self.assertEqual(r.outcomes["preempted"].gpu_seconds, 2 * H)
        self.assertEqual(r.outcomes["completed"].gpu_seconds, 4 * H)
        assert_sums(self, r)

    def test_job_listing_merges_attempts_without_adding_restarts(self):
        _, h = loaded(self.jobs())
        page = h.jobs(Window(T0, T0 + DAY))
        first = [row for row in page.rows if row.job_id == "1"][0]
        self.assertEqual(page.total, 3)
        self.assertEqual(first.attempts, 3)
        self.assertEqual(first.gpu_seconds, 4 * H)
        self.assertEqual(first.restarts, 2)
        self.assertEqual(first.first_start, T0)
        self.assertEqual(len(h.jobs(Window(T0, T0 + DAY), by="attempt").rows), 5)
        self.assertEqual([a.state for a in h.attempts("1")], ["PREEMPTED", "PREEMPTED",
                                                               "COMPLETED"])

    def test_job_id_reuse_by_another_user_is_another_job(self):
        jobs = [job(7, start=T0, end=T0 + H, uid=1, user="a", index=1),
                job(7, start=T0 + H, end=T0 + 2 * H, uid=2, user="b", index=2)]
        _, h = loaded(jobs)
        self.assertEqual(h.query(Window(T0, T0 + DAY)).total.jobs, 2)

    def test_outcome_names_never_say_wasted(self):
        self.assertIn("preempted", OUTCOMES)
        self.assertNotIn("wasted", OUTCOMES)


class GroupingTest(unittest.TestCase):
    def jobs(self):
        return [
            job(1, start=T0, end=T0 + 5 * H, user="a", uid=1, account="x", qos="high", gpus=2),
            job(2, start=T0 + H, end=T0 + 3 * H, user="b", uid=2, account="y", qos="low",
                gpus=4, nodes="g[01-02]", cpus=8, alloc="cpu=8,gres/gpu=4,node=2", state="FAILED"),
            job(3, start=T0 + 2 * H, end=T0 + 9 * H, user="a", uid=1, account="y",
                partition="cpu", gpus=0, cpus=16),
        ]

    def test_every_grouping_sums_to_total(self):
        _, h = loaded(self.jobs())
        r = h.query(Window(T0, T0 + DAY), group_by=["user", "account", "qos", "partition",
                                                    "state", ("user", "qos"),
                                                    ("account", "user"), "node"])
        assert_sums(self, r)
        self.assertEqual(r.group("user")[("a",)].gpu_seconds, 10 * H)
        self.assertEqual(r.group("user", "qos")[("b", "low")].gpu_seconds, 8 * H)
        node = r.group("node")
        self.assertEqual(sum(s.gpu_seconds for s in node.values()), r.total.gpu_seconds)
        self.assertEqual(node[("g02",)].gpu_seconds, 2 * 2 * H)
        self.assertEqual(r.ranked("user")[0][0], ("a",))

    def test_peak_and_active_time(self):
        _, h = loaded(self.jobs())
        t = h.query(Window(T0, T0 + DAY)).total
        self.assertEqual(t.peak_gpus, 6)
        self.assertEqual(t.active_seconds, 9 * H)
        self.assertEqual(t.gpu_active_seconds, 5 * H)
        self.assertEqual((t.first, t.last), (T0, T0 + 9 * H))

    def test_filters(self):
        _, h = loaded(self.jobs())
        w = Window(T0, T0 + DAY)
        self.assertEqual(h.query(w, Filter(user="a")).total.cpu_seconds, 4 * 5 * H + 16 * 7 * H)
        self.assertEqual(h.query(w, Filter(account=["x", "y"], qos="low")).total.attempts, 1)
        self.assertEqual(h.query(w, Filter(outcome="failed")).total.gpu_seconds, 8 * H)
        self.assertEqual(h.query(w, Filter(state="COMPLETED")).total.attempts, 2)
        on_node = h.query(w, Filter(node="g02")).total
        self.assertEqual(on_node.gpu_seconds, 2 * 2 * H)
        self.assertEqual(on_node.cpu_seconds, 4 * 2 * H)
        self.assertEqual(h.query(w, Filter(user="nobody")).total.attempts, 0)
        with self.assertRaises(ValueError):
            Filter(outcome="wasted")

    def test_series_and_lanes(self):
        _, h = loaded(self.jobs())
        r = h.query(Window(T0, T0 + DAY), series_by="user", lanes_by="user", lane_resolution=0)
        self.assertEqual(sum(r.buckets("gpu", key="a")[i].value for i in range(24)), 10 * H)
        hours = r.buckets("gpu")
        self.assertEqual(len(hours), 24)
        self.assertEqual(hours[1].value, (2 + 4) * H)
        lanes = {lane.key: lane.intervals for lane in r.lanes}
        self.assertEqual(lanes["a"], [(T0, T0 + 9 * H)])

    def test_waits_and_time_limit_accuracy(self):
        jobs = [job(1, submit=T0, start=T0 + H, end=T0 + 2 * H, limit=120),
                job(2, submit=T0, eligible=T0 + 2 * H, start=T0 + 5 * H, end=T0 + 6 * H,
                    limit=60, state="TIMEOUT")]
        _, h = loaded(jobs)
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual(list(r.waits), [H, 3 * H])
        self.assertEqual(r.total.wait_max, 3 * H)
        self.assertEqual(r.wait_quantile(0.5), H)
        self.assertAlmostEqual(r.total.time_limit_used, 2 * H / (3 * H))


class UnknownTest(unittest.TestCase):
    def test_gpus_not_tracked_is_none_not_zero(self):
        config = FakeSlurm().config.replace(",gres/gpu", "")
        _, h = loaded([job(1, start=T0, end=T0 + H, gpus=0)], config=config)
        r = h.query(Window(T0, T0 + DAY))
        self.assertIsNone(r.total.gpu_hours)
        self.assertIsNone(r.buckets("gpu"))
        self.assertTrue(any(n.code == "no_gpu_tres" for n in r.notes))

    def test_unknown_memory_is_not_zero(self):
        _, h = loaded([job(1, start=T0, end=T0 + H, req="cpu=4,node=1", req_mem="0")])
        t = h.query(Window(T0, T0 + DAY)).total
        self.assertIsNone(t.mem_gib_hours)
        self.assertFalse(t.complete("mem"))

    def test_query_before_loading_raises(self):
        h = make_history(FakeSlurm([]))
        with self.assertRaises(HistoryError):
            h.query()


class DstTest(unittest.TestCase):
    def test_spring_forward_day_has_23_hours_and_nothing_is_lost(self):
        with timezone("America/New_York"):
            import datetime
            day = datetime.date(2030, 3, 10)
            w = Window.days(day)
            self.assertEqual(w.end - w.start, 23 * H)
            fake = FakeSlurm([job(1, start=w.start - H, end=w.end + H, gpus=1)],
                             now=w.end + DAY, live=[])
            h = make_history(fake)
            h.refresh()
            r = h.query(w)
            self.assertEqual(len(r.buckets("gpu", "hour")), 23)
            self.assertEqual(r.total.gpu_seconds, 23 * H)
            days = h.query(Window(w.start - DAY, w.end + DAY)).buckets("gpu", "day")
            self.assertEqual([b.end - b.start for b in days], [24 * H, 23 * H, 24 * H])
            assert_sums(self, r)

    def test_fall_back_day_has_25_hours(self):
        with timezone("America/New_York"):
            import datetime
            w = Window.days(datetime.date(2030, 11, 3))
            fake = FakeSlurm([job(1, start=w.start, end=w.end, gpus=2)], now=w.end + DAY,
                             live=[])
            h = make_history(fake)
            h.refresh()
            r = h.query(w)
            self.assertEqual(len(r.buckets("gpu")), 25)
            self.assertEqual(r.total.gpu_seconds, 2 * 25 * H)
            self.assertEqual(sum(map(sum, r.heatmap("gpu"))), 2 * 25 * H)

    def test_half_hour_zone(self):
        with timezone("Asia/Kolkata"):
            _, h = loaded([job(1, start=T0, end=T0 + 5 * H, gpus=1)])
            r = h.query(Window(T0, T0 + DAY))
            assert_sums(self, r)
            import time
            self.assertTrue(all(time.localtime(b.start).tm_min == 0
                                for b in r.buckets("gpu")[1:]))


class UtilizationTest(unittest.TestCase):
    def test_capacity_mismatch_and_live_capacity(self):
        _, h = loaded([job(1, start=T0, end=T0 + DAY, gpus=8)], cluster_tres="cpu=64,gres/gpu=32")
        checks = {c.metric: c for c in h.capacity()}
        self.assertEqual((checks["gres/gpu"].live, checks["gres/gpu"].accounting), (16, 32))
        self.assertTrue(checks["gres/gpu"].mismatch)
        r = h.query(Window(T0, T0 + DAY))
        self.assertAlmostEqual(r.utilization(checks["gres/gpu"].live).fraction, 0.5)
        self.assertIn("today", r.utilization(16).label)


if __name__ == "__main__":
    unittest.main()
