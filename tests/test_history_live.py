"""Read-only checks against whatever cluster this runs on; skipped without accounting.

Uses a throwaway cache and a short window. Fetches no scripts, submit lines or
environments.
"""

import shutil
import subprocess
import tempfile
import time
import unittest

from slurm_usage.history import History, HistoryError, Status, Window


def accounting_up() -> bool:
    if shutil.which("sacct") is None:
        return False
    try:
        return subprocess.run(["sacct", "-n", "-X", "-S", "now-1hours", "-o", "JobIDRaw"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(accounting_up(), "no Slurm accounting reachable")
class LiveHistoryTest(unittest.TestCase):
    def test_last_day_is_consistent(self):
        directory = tempfile.mkdtemp(prefix="history-live-")
        self.addCleanup(shutil.rmtree, directory, True)
        history = History(cache_dir=directory)
        info = history.info()
        if info.accounting == Status.UNAVAILABLE:
            self.skipTest("accounting unavailable: %s" % info.notes)
        window = Window.last(86400)
        started = time.time()
        try:
            history.refresh(window)
        except HistoryError as error:
            self.skipTest("accounting refused: %s" % error)
        loaded = time.time() - started
        report = history.query(window, group_by=["user", "account", "qos", "partition",
                                                 ("user", "qos")])
        total = report.total
        for spec, groups in report.groups.items():
            self.assertEqual(sum(s.cpu_seconds for s in groups.values()), total.cpu_seconds, spec)
            self.assertEqual(sum(s.gpu_seconds for s in groups.values()), total.gpu_seconds, spec)
        if total.amount("gpu") is not None:
            self.assertEqual(sum(b.value for b in report.buckets("gpu", "hour")),
                             total.gpu_seconds)
        self.assertTrue(report.complete)
        history.build_rollup()
        day = history.overview().ranges[0]
        self.assertEqual(day.name, "day")
        print("\nhistory: %d attempts in the last day, loaded in %.1fs, visibility %s, notes %s"
              % (total.attempts, loaded, info.visibility, [n.code for n in report.notes]))
        history.close()


if __name__ == "__main__":
    unittest.main()
