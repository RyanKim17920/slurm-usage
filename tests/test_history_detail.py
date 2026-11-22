"""Efficiency, one job's detail, log paths, scripts and submit lines."""

import unittest

from tests.test_history_support import H, T0, FakeSlurm, job, make_history, step

from slurm_usage.history import Status, Window
from slurm_usage.history.detail import fetch_batch_script
from slurm_usage.history.efficiency import AttemptUsage
from slurm_usage.history.paths import expand_path
from slurm_usage.runner import Result

DAY = 86400
GIB = 1024 ** 3


def gpu_job(job_id, util, gpus=4, cpus=8, mem="64G", total_cpu="16:00:00", rss="32G", **kw):
    return job(job_id, start=T0, end=T0 + 4 * H, gpus=gpus, cpus=cpus, mem=mem,
               TotalCPU=total_cpu, steps=[
                   step("extern", TRESUsageInTot="cpu=00:00:00,mem=0"),
                   step("batch", TotalCPU=total_cpu,
                        TRESUsageInTot="cpu=%s,mem=%s,gres/gpuutil=%s" % (total_cpu, rss, util))],
               **kw)


class EfficiencyTest(unittest.TestCase):
    def test_efficiency_with_coverage(self):
        fake = FakeSlurm([gpu_job(1, 200), job(2, start=T0, end=T0 + 4 * H, gpus=4, cpus=8)],
                         live=[])
        h = make_history(fake)
        h.refresh()
        before = h.query(Window(T0, T0 + DAY), efficiency=True)
        self.assertEqual(before.efficiency_unchecked, 2)
        self.assertIsNone(before.total.efficiency("gpu").value)
        self.assertEqual(h.load_efficiency(Window(T0, T0 + DAY)), 2)
        r = h.query(Window(T0, T0 + DAY), efficiency=True)
        self.assertEqual(r.efficiency_unchecked, 0)
        gpu = r.total.efficiency("gpu")
        self.assertAlmostEqual(gpu.value, 0.5)  # 200 summed over 4 GPUs
        self.assertAlmostEqual(gpu.coverage, 0.5)  # job 2 has no samples: not zero
        self.assertAlmostEqual(r.total.efficiency("cpu").value, 0.5)
        self.assertAlmostEqual(r.total.efficiency("mem").value, 0.5)
        self.assertEqual(h.load_efficiency(Window(T0, T0 + DAY)), 0)  # remembered

    def test_values_above_100_per_gpu_are_capped_and_counted(self):
        fake = FakeSlurm([gpu_job(1, 900, gpus=4)], live=[])
        h = make_history(fake)
        h.refresh()
        h.load_efficiency()
        total = h.query(efficiency=True).total
        self.assertEqual(total.efficiency("gpu").value, 1.0)
        self.assertEqual(total.gpu_over_100, 1)

    def test_missing_samples_are_none_not_zero(self):
        usage = AttemptUsage("")
        usage.step("", "cpu=00:00:00,mem=0,gres/gpuutil=0")
        self.assertEqual(usage.result(), (None, None, None))

    def test_running_jobs_are_not_read(self):
        fake = FakeSlurm([job(1, start=T0, end=None, state="RUNNING")], now=T0 + H, live=[1])
        h = make_history(fake)
        h.refresh()
        self.assertEqual(h.load_efficiency(), 0)

    def test_scoped_to_one_user(self):
        from slurm_usage.history import Filter
        fake = FakeSlurm([gpu_job(1, 100, user="a", uid=7)], live=[])
        h = make_history(fake)
        h.refresh()
        h.load_efficiency(where=Filter(user="a"))
        self.assertTrue(fake.sacct_calls("-u", "7"))


class DetailTest(unittest.TestCase):
    def test_all_attempts_and_steps(self):
        jobs = [job(4, start=T0, end=T0 + H, state="PREEMPTED", index=1, submit=T0,
                    ExitCode="0:15"),
                gpu_job(4, 100, index=2, submit=T0 + H, StdOut="/logs/%x-%A_%a.out",
                        JobID="4", name="sweep")]
        fake = FakeSlurm(jobs, live=[])
        h = make_history(fake)
        h.refresh()
        detail = h.detail(4)
        self.assertEqual(detail.status, Status.OK)
        self.assertEqual([a.job.state for a in detail.attempts], ["PREEMPTED", "COMPLETED"])
        self.assertEqual(detail.attempts[0].exit_signal, 15)
        last = detail.attempts[1]
        self.assertEqual([s.step_id for s in last.steps], ["extern", "batch"])
        self.assertEqual(last.steps[1].gpu_util_total, 100)
        self.assertIsNone(last.steps[0].max_rss)
        self.assertEqual(last.stdout.display, "/logs/sweep-4_%a.out")
        self.assertFalse(last.stdout.resolved)
        self.assertTrue(last.stderr.merged)
        self.assertAlmostEqual(last.job.gpu_peak_utilization, 0.25)

    def test_raw_workdir_is_exact_display_is_clean(self):
        fake = FakeSlurm([job(6, start=T0, end=T0 + H, WorkDir="/work/a\x1b[2Jb ")], live=[])
        h = make_history(fake)
        h.refresh()
        last = h.detail(6).attempts[-1]
        self.assertEqual(last.raw_workdir, "/work/a\x1b[2Jb ")
        self.assertNotIn("\x1b", last.workdir)
        self.assertNotIn("raw_workdir", repr(last))

    def test_unknown_job(self):
        fake = FakeSlurm([], live=[])
        h = make_history(fake)
        detail = h.detail(99)
        self.assertEqual(detail.status, Status.EMPTY)
        with self.assertRaises(ValueError):
            h.detail("1;rm")

    def test_detail_refused(self):
        fake = FakeSlurm([job(1, start=T0, end=T0 + H)], live=[])
        fake.fail["-j 5"] = Status.DENIED
        self.assertEqual(make_history(fake).detail(5).status, Status.DENIED)


class PathTest(unittest.TestCase):
    def test_patterns(self):
        path = expand_path("out/%x_%A_%a_%j_%u_%N_%3a.log", "101", "100_7", "alice", "run",
                           "g[01-02]", "/home/alice")
        self.assertEqual(path.display, "/home/alice/out/run_100_7_101_alice_g01_007.log")
        self.assertTrue(path.resolved)
        self.assertEqual(expand_path("%%j", "1", "1", "u", "n", "", "/w").display, "/w/%j")
        self.assertIsNone(expand_path("", "1", "1", "u", "n", "", "/w"))

    def test_hostile_path_is_cleaned_for_display_only(self):
        path = expand_path("/tmp/\x1b[31mx.log", "1", "1", "u", "n", "", "/w")
        self.assertNotIn("\x1b", path.display)
        self.assertIn("\x1b", path.raw)
        self.assertNotIn("\x1b", repr(path))


class OnDemandTest(unittest.TestCase):
    def test_script_allowed(self):
        fake = FakeSlurm()
        fake.scripts["3"] = Result(["sacct"], Status.OK,
                                   "Batch Script for 3\n----------\n#!/bin/sh\necho hi\n", "", 0)
        text = make_history(fake).batch_script(3)
        self.assertEqual((text.status, text.text), (Status.OK, "#!/bin/sh\necho hi"))
        self.assertNotIn("echo", repr(text))

    def test_script_denied_and_absent(self):
        fake = FakeSlurm()
        fake.scripts["3"] = Result(["sacct"], Status.DENIED, "", "Permission denied", 1,
                                   "Permission denied")
        self.assertEqual(make_history(fake).batch_script(3).status, Status.DENIED)
        fake.scripts["3"] = Result(["sacct"], Status.OK, "Batch Script for 3\n---\nNONE\n", "", 0)
        self.assertEqual(make_history(fake).batch_script(3).status, Status.NOT_STORED)

    def test_script_header_only_bytes_and_no_record(self):
        fake = FakeSlurm()
        raw = (b"Batch Script for 3\n" + b"-" * 80 + b"\nNONE\n").decode()
        fake.scripts["3"] = Result(["sacct"], Status.OK, raw, "", 0)
        text = make_history(fake).batch_script(3)
        self.assertEqual((text.status, text.text), (Status.NOT_STORED, ""))
        self.assertIn("does not store scripts", text.message)
        fake.scripts["3"] = Result(["sacct"], Status.EMPTY, "", "", 0)
        text = make_history(fake).batch_script(3)
        self.assertEqual(text.status, Status.DENIED)
        self.assertIn("owner", text.message)
        fake.scripts["3"] = Result(["sacct"], Status.PARTIAL, "Batch Script for 3\n---\n",
                                   "sacct: error: Permission denied", 1, "Permission denied")
        self.assertEqual(make_history(fake).batch_script(3).status, Status.DENIED)

    def test_script_and_submit_line_are_never_cached(self):
        fake = FakeSlurm([job(3, start=T0, end=T0 + H, SubmitLine="sbatch --x\n")], live=[])
        h = make_history(fake)
        h.refresh()
        self.assertEqual(h.submit_line(3).text, "sbatch --x\n")
        h.submit_line(3)
        self.assertEqual(len(fake.sacct_calls("-j", "3")), 2)
        fake.fail["SubmitLine"] = Status.DENIED
        self.assertEqual(h.submit_line(3).status, Status.DENIED)
        with self.assertRaises(ValueError):
            fetch_batch_script(fake, "3_1")


if __name__ == "__main__":
    unittest.main()
