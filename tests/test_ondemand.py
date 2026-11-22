import os
import shutil
import tempfile
import threading
import unittest

from slurm_usage.live import ondemand, tail_file
from slurm_usage.model import Job, Status
from slurm_usage.runner import Result
from tests import cluster_fixture as cf
from tests import fixtures as fx

SSTAT = (
    "42.extern|1|213503982334-14:25:51|0|0|energy=0|energy=0|energy=0|gpu01\n"
    "42.batch|1|00:10:00|1000K|2000K|cpu=00:10:00,mem=1000K,gres/gpuutil=150|"
    "cpu=00:10:00,mem=2000K,gres/gpuutil=150,gres/gpumem=1G|"
    "cpu=00:10:00,mem=1000K,gres/gpuutil=150,gres/gpumem=1G|gpu01\n"
    "42.0|2|99999-00:00:00|5M|8M|cpu=1:00|cpu=1:00|cpu=2:00|gpu[01-02]\n"
    "43.batch|1|00:00:01|1K|1K|cpu=1|cpu=1|cpu=1|gpu09\n")


def running(job_id="42", uid=cf.ALICE, **extra):
    job = Job(job_id=job_id, display_id=job_id, state="RUNNING", uid=uid, user="alice",
              elapsed=3600, cpus_alloc=8, nodes=["gpu01"], batch_host="gpu01",
              raw_name="train", raw_workdir="/work/alice")
    job.__dict__.update(extra)
    return job


class StepsTest(unittest.TestCase):
    def test_parse_and_garbage(self):
        collector = cf.collector({"sstat": SSTAT})
        report = collector.steps(running())
        self.assertEqual(report.status, Status.OK)
        steps = {step.step_id: step for step in report.steps}
        self.assertEqual(sorted(steps), ["0", "batch", "extern"])
        self.assertIsNone(steps["extern"].ave_cpu)
        self.assertEqual(steps["batch"].ave_cpu, 600)
        self.assertEqual(steps["batch"].max_rss, 2000 * 1024)
        self.assertEqual(steps["batch"].gpu_util_ave, 150)
        self.assertEqual(steps["batch"].gpu_mem_max, 2**30)
        self.assertTrue(any("not percent busy" in note for note in steps["batch"].notes))
        self.assertIsNone(steps["0"].ave_cpu)
        self.assertIn("impossible CPU time reported", steps["0"].notes)
        self.assertEqual(steps["0"].nodes, ["gpu01", "gpu02"])
        self.assertEqual(collector.runner.calls[-1][:4], ["sstat", "-a", "-n", "-P"])

    def test_someone_elses_job_is_attempted_then_remembered(self):
        refusal = fx.denied("Invalid user id")
        collector = cf.collector({"sstat": refusal})
        job = running("77", uid=cf.BOB)
        self.assertEqual(collector.steps(job).status, Status.DENIED)
        self.assertEqual(collector.steps(job).status, Status.DENIED)
        self.assertEqual(sum(argv[0] == "sstat" for argv in collector.runner.calls), 1)

    def test_allowed_for_someone_else_when_slurm_allows(self):
        collector = cf.collector({"sstat": SSTAT.replace("42", "77")})
        self.assertEqual(collector.steps(running("77", uid=cf.BOB)).status, Status.OK)

    def test_not_running(self):
        collector = cf.collector({})
        report = collector.steps(running(state="PENDING"))
        self.assertEqual(report.status, Status.EMPTY)
        self.assertEqual(collector.runner.calls, [])


class FetchTest(unittest.TestCase):
    def test_batch_script_sanitized(self):
        text = "Batch Script for 42\n" + "-" * 20 + "\n#!/bin/bash\necho \x1b]0;t\x07hi\x1b[2J\n"
        fetched = cf.collector({"sacct script": text}).batch_script(running())
        self.assertEqual(fetched.status, Status.OK)
        self.assertNotIn("\x1b", fetched.text)
        self.assertIn("echo hi", fetched.text)

    def test_batch_script_denied_and_none(self):
        self.assertEqual(cf.collector({"sacct script": fx.denied()}).batch_script(
            running("88", uid=cf.BOB)).status, Status.DENIED)
        none = cf.collector({"sacct script": "Batch Script for 42\n----\nNONE\n"})
        self.assertEqual(none.batch_script(running()).status, Status.NOT_STORED)

    def test_batch_script_header_only_bytes(self):
        # exactly what sacct prints when the record exists but no script was kept
        header = b"Batch Script for 42\n" + b"-" * 80 + b"\nNONE\n"
        fetched = cf.collector({"sacct script": header.decode()}).batch_script(running())
        self.assertEqual(fetched.status, Status.NOT_STORED)
        self.assertIn("AccountingStoreFlags", fetched.detail)
        self.assertEqual(fetched.text, "")

    def test_batch_script_no_record_is_denied(self):
        # sacct prints nothing, exit 0, for a job it will not return to this invoker
        fetched = cf.collector({"sacct script": ""}).batch_script(running("88", uid=cf.BOB))
        self.assertEqual(fetched.status, Status.DENIED)
        self.assertIn("owner", fetched.detail)

    def test_batch_script_is_not_cached(self):
        collector = cf.collector({"sacct script": "#!/bin/sh\n"})
        collector.batch_script(running())
        collector.batch_script(running())
        self.assertEqual(len(collector.runner.calls), 2)

    def test_submit_line(self):
        def answer(argv):
            delimiter = [a for a in argv if a.startswith("--delimiter=")][0][12:]
            return ("42%ssbatch --wrap='a\x1b[1m|b'\n--more\n42.batch%s\n"
                    % (delimiter, delimiter))
        fetched = cf.collector({"sacct submit": answer}).submit_line(running())
        self.assertEqual(fetched.status, Status.OK)
        self.assertEqual(fetched.text, "sbatch --wrap='a|b'\n--more")

    def test_bad_job_id_never_queried(self):
        collector = cf.collector({})
        self.assertEqual(collector.submit_line(running("1,2")).status, Status.EMPTY)
        self.assertEqual(collector.batch_script(running("-a")).status, Status.EMPTY)
        self.assertEqual(collector.runner.calls, [])


class LogPathTest(unittest.TestCase):
    def answer(self, out, err=""):
        def respond(argv):
            spec = argv[argv.index("-O") + 1]
            sep = spec.split("JobID:")[1].split(",")[0]
            end = spec.rsplit("STDERR:", 1)[1]
            return "42" + sep + out + sep + err + end + "\n"
        return respond

    def test_patterns(self):
        job = running(array_job_id="40", array_task="2", user="alice")
        paths = cf.collector({"squeue log": self.answer("logs/%x_%A_%a_%3j.%N.out")}
                             ).log_paths(job)
        self.assertEqual(paths.status, Status.OK)
        self.assertEqual(paths.raw_stdout, "/work/alice/logs/train_40_2_042.gpu01.out")
        self.assertEqual(paths.raw_stderr, paths.raw_stdout)

    def test_unknown_pattern_and_escape(self):
        paths = cf.collector({"squeue log": self.answer("/l/%J.out", "/l/\\%j.err")}
                             ).log_paths(running())
        self.assertEqual(paths.status, Status.PARTIAL)
        self.assertEqual(paths.unresolved, ["%J"])
        self.assertEqual(paths.raw_stderr, "/l/%j.err")

    def test_hostile_path_sanitized_for_display(self):
        paths = cf.collector({"squeue log": self.answer("/l/\x1b[2Jx.out")}).log_paths(running())
        self.assertNotIn("\x1b", paths.stdout)
        self.assertIn("\x1b", paths.raw_stdout)

    def test_gone_and_refused(self):
        gone = Result([], Status.EMPTY, "", "slurm_load_jobs error: Invalid job id", 1,
                      "Invalid job id")
        self.assertEqual(cf.collector({"squeue log": gone}).log_paths(running()).status,
                         Status.EMPTY)
        self.assertEqual(cf.collector({"squeue log": fx.denied()}).log_paths(running()).status,
                         Status.DENIED)


class TailTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="slurm_usage_logs_")
        self.uid = os.getuid()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write(self, name, data):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def test_tail_bounded_and_sanitized(self):
        lines = b"".join(b"line %d \x1b[31mred\x1b[0m\n" % i for i in range(1000))
        path = self.write("a.out", lines + b"epoch 1\rprogress 50%\rprogress 100%\n")
        tail = tail_file(path, self.uid, max_bytes=200)
        self.assertEqual(tail.status, Status.OK)
        self.assertTrue(tail.truncated)
        self.assertEqual(tail.lines[-1], "progress 100%")
        self.assertTrue(all("\x1b" not in line for line in tail.lines))
        self.assertTrue(tail.lines[0].startswith("line "))  # the cut first line is dropped

    def test_fifo_does_not_hang(self):
        path = os.path.join(self.dir, "pipe")
        os.mkfifo(path)
        done = []
        worker = threading.Thread(target=lambda: done.append(tail_file(path, self.uid)))
        worker.daemon = True
        worker.start()
        worker.join(5)
        self.assertTrue(done, "tail_file hung on a FIFO")
        self.assertEqual(done[0].status, Status.DENIED)

    def test_device_and_pseudo_files_refused(self):
        self.assertEqual(tail_file("/dev/zero", self.uid).status, Status.DENIED)
        self.assertEqual(tail_file("/proc/self/environ", self.uid).status, Status.DENIED)

    def test_symlink_to_pseudo_file_refused(self):
        link = os.path.join(self.dir, "link.out")
        os.symlink("/proc/self/status", link)
        self.assertEqual(tail_file(link, self.uid).status, Status.DENIED)

    def test_own_file_through_someone_elses_job_refused(self):
        path = self.write("mine.txt", b"private\n")
        tail = tail_file(path, self.uid + 12345)
        self.assertEqual(tail.status, Status.DENIED)
        self.assertNotIn("private", " ".join(tail.lines))

    def test_missing_relative_and_empty(self):
        self.assertEqual(tail_file(os.path.join(self.dir, "none"), self.uid).status, Status.EMPTY)
        self.assertEqual(tail_file("relative.out", self.uid).status, Status.EMPTY)
        self.assertEqual(tail_file(self.write("e", b""), self.uid).status, Status.EMPTY)

    def test_directory_refused(self):
        self.assertEqual(tail_file(self.dir, self.uid).status, Status.DENIED)


if __name__ == "__main__":
    unittest.main()
