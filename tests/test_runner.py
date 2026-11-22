"""The subprocess path, against fake Slurm executables in a temporary directory."""

import os
import shutil
import stat
import sys
import tempfile
import textwrap
import time
import unittest

from slurm_usage.model import Status
from slurm_usage.runner import ReadOnlyViolation, Runner, classify


def make_tool(directory, name, body):
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write("#!%s -B\nimport os, sys, time\n%s\n" % (sys.executable,
                                                              textwrap.dedent(body)))
    os.chmod(path, stat.S_IRWXU)
    return path


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="slurm_usage_test_")
        environ = {"PATH": self.dir, "HOME": self.dir, "SQUEUE_FORMAT": "%i",
                   "SLURM_CLUSTERS": "other", "LANG": "xx_XX.UTF-8"}
        self.runner = Runner(timeout=5, search_path=self.dir, environ=environ)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_ok_and_environment(self):
        make_tool(self.dir, "squeue", """
            for key in ("LC_ALL", "SLURM_TIME_FORMAT", "SLURM_BITSTR_LEN", "SQUEUE_FORMAT",
                        "SLURM_CLUSTERS", "HOME"):
                print(key, os.environ.get(key))
            print("argv", sys.argv[1:])
        """)
        result = self.runner.run(["squeue", "-h", "-o", "%i"])
        self.assertEqual(result.status, Status.OK)
        lines = dict(line.split(" ", 1) for line in result.lines())
        self.assertEqual(lines["LC_ALL"], "C")
        self.assertEqual(lines["SLURM_TIME_FORMAT"], "%s")
        self.assertEqual(lines["SLURM_BITSTR_LEN"], "0")
        self.assertEqual(lines["SQUEUE_FORMAT"], "None")
        self.assertEqual(lines["SLURM_CLUSTERS"], "None")
        self.assertEqual(lines["HOME"], self.dir)
        self.assertEqual(lines["argv"], "['-h', '-o', '%i']")

    def test_arguments_are_not_shell_interpreted(self):
        make_tool(self.dir, "squeue", "print(sys.argv[1:])")
        result = self.runner.run(["squeue", "-o", "$(touch pwned); `id` | x"])
        self.assertIn("$(touch pwned); `id` | x", result.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "pwned")))

    def test_missing_program(self):
        result = self.runner.run(["sstat", "-j", "1"])
        self.assertEqual(result.status, Status.UNAVAILABLE)
        self.assertFalse(self.runner.has("sstat"))

    def test_relative_path_entries_ignored(self):
        make_tool(self.dir, "squeue", "print('hijacked')")
        runner = Runner(search_path=".:relative", environ={"PATH": "."})
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            self.assertEqual(runner.run(["squeue"]).status, Status.UNAVAILABLE)
        finally:
            os.chdir(cwd)

    def test_policy_runs_before_anything(self):
        make_tool(self.dir, "scontrol", "open(os.path.join(%r, 'ran'), 'w')" % self.dir)
        with self.assertRaises(ReadOnlyViolation):
            self.runner.run(["scontrol", "update", "JobId=1", "Priority=0"])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "ran")))

    def test_denied(self):
        make_tool(self.dir, "sstat", """
            sys.stderr.write("sstat: error: problem getting step_layout for StepId=7.batch: "
                             "Invalid user id\\n")
        """)
        result = self.runner.run(["sstat", "-a", "-j", "7"])
        self.assertEqual(result.status, Status.DENIED)
        self.assertIn("Invalid user id", result.reason)

    def test_accounting_unavailable(self):
        make_tool(self.dir, "sacct", """
            sys.stderr.write("sacct: error: Slurm accounting storage is disabled\\n")
            sys.exit(1)
        """)
        self.assertEqual(self.runner.run(["sacct", "-n"]).status, Status.UNAVAILABLE)

    def test_controller_down(self):
        make_tool(self.dir, "squeue", """
            sys.stderr.write("slurm_load_jobs error: Unable to contact slurm controller\\n")
            sys.exit(1)
        """)
        self.assertEqual(self.runner.run(["squeue"]).status, Status.UNAVAILABLE)

    def test_empty_and_invalid_job(self):
        make_tool(self.dir, "squeue", "sys.exit(0)")
        self.assertEqual(self.runner.run(["squeue"]).status, Status.EMPTY)
        make_tool(self.dir, "sstat", """
            sys.stderr.write("sstat: error: couldn't get steps for JobId=9\\n")
        """)
        self.assertEqual(self.runner.run(["sstat", "-j", "9"]).status, Status.EMPTY)

    def test_error(self):
        make_tool(self.dir, "sinfo", "sys.stderr.write('sinfo: fatal: boom\\n'); sys.exit(2)")
        result = self.runner.run(["sinfo"])
        self.assertEqual(result.status, Status.ERROR)
        self.assertEqual(result.returncode, 2)

    def test_partial(self):
        make_tool(self.dir, "squeue", """
            print("1")
            sys.stderr.write("squeue: error: some records skipped\\n")
        """)
        self.assertEqual(self.runner.run(["squeue"]).status, Status.PARTIAL)

    def test_stdout_cannot_fake_a_status(self):
        make_tool(self.dir, "squeue", "print('Permission denied: Unable to contact slurm')")
        self.assertEqual(self.runner.run(["squeue"]).status, Status.OK)

    def test_hostile_reason_is_sanitized(self):
        make_tool(self.dir, "squeue", r"""
            sys.stderr.write("error: \x1b]0;title\x07bad\x1b[2J\n"); sys.exit(1)
        """)
        self.assertNotIn("\x1b", self.runner.run(["squeue"]).reason)

    def test_timeout_kills_process_group(self):
        make_tool(self.dir, "squeue", """
            import subprocess
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            time.sleep(60)
        """)
        started = time.monotonic()
        result = self.runner.run(["squeue"], timeout=0.5)
        self.assertEqual(result.status, Status.TIMEOUT)
        self.assertLess(time.monotonic() - started, 10)

    def test_stream_and_cancel(self):
        make_tool(self.dir, "sacct", """
            for number in range(10**7):
                print("line", number, flush=number < 5)
        """)
        started = time.monotonic()
        with self.runner.stream(["sacct", "-n"]) as rows:
            seen = []
            for line in rows:
                seen.append(line)
                if len(seen) == 3:
                    break
        self.assertEqual(seen, ["line 0", "line 1", "line 2"])
        self.assertEqual(rows.result.status, Status.CANCELLED)
        self.assertLess(time.monotonic() - started, 10)

    def test_stream_complete(self):
        make_tool(self.dir, "sacct", "print('a'); print('b')")
        with self.runner.stream(["sacct", "-n"]) as rows:
            self.assertEqual(list(rows), ["a", "b"])
        self.assertEqual(rows.result.status, Status.OK)

    def test_output_limit(self):
        make_tool(self.dir, "sacct", "sys.stdout.write('x' * 100000 + '\\n')")
        result = self.runner.run(["sacct", "-n"], max_bytes=1000)
        self.assertEqual(result.status, Status.PARTIAL)
        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.stdout), 1001)

    def test_undecodable_output_survives(self):
        make_tool(self.dir, "squeue", "sys.stdout.buffer.write(b'name\\xff\\xfe\\n')")
        self.assertEqual(self.runner.run(["squeue"]).lines()[0][:4], "name")

    def test_lines_split_only_on_newline(self):
        make_tool(self.dir, "squeue", "sys.stdout.write('a\\rb\\x1cc\\u2028d\\n')")
        self.assertEqual(len(self.runner.run(["squeue"]).lines()), 1)

    def test_batch_script_allowed_and_denied(self):
        make_tool(self.dir, "sacct", """
            if sys.argv[1:] != ["-j", "42", "--batch-script"]:
                sys.exit(3)
            print("Batch Script for 42\\n----\\n#!/bin/bash\\necho \\x1b[31mhi")
        """)
        self.assertEqual(self.runner.batch_script("42").status, Status.OK)
        make_tool(self.dir, "sacct", """
            sys.stderr.write("sacct: error: You do not have permission to view this job\\n")
            sys.exit(1)
        """)
        self.assertEqual(self.runner.batch_script("43").status, Status.DENIED)

    def test_batch_script_header_only_and_no_record_bytes(self):
        from slurm_usage.history.detail import fetch_batch_script as history_script
        from slurm_usage.live.ondemand import fetch_batch_script as live_script
        # the exact bytes sacct writes when the record exists but holds no script
        make_tool(self.dir, "sacct", """
            sys.stdout.buffer.write(b"Batch Script for 42\\n" + b"-" * 80 + b"\\nNONE\\n")
        """)
        self.assertEqual(live_script(self.runner, "42").status, Status.NOT_STORED)
        self.assertEqual(history_script(self.runner, "42").status, Status.NOT_STORED)
        # and when it returns no record to this invoker: nothing at all, exit 0
        make_tool(self.dir, "sacct", "pass")
        self.assertEqual(live_script(self.runner, "42").status, Status.DENIED)
        self.assertEqual(history_script(self.runner, "42").status, Status.DENIED)

    def test_classify_table(self):
        self.assertEqual(classify(0, True, ""), Status.OK)
        self.assertEqual(classify(1, True, ""), Status.PARTIAL)
        self.assertEqual(classify(0, False, ""), Status.EMPTY)
        self.assertEqual(classify(1, False, "Access denied"), Status.DENIED)
        self.assertEqual(classify(1, False, ""), Status.ERROR)


if __name__ == "__main__":
    unittest.main()
