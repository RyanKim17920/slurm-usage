"""Read-only checks against whatever Slurm cluster this runs on; skipped without one.

Runs only the collector's normal bulk queries. Prints a short summary.
"""

import shutil
import subprocess
import sys
import unittest

from slurm_usage.live import LiveCollector, reconcile_gpus
from slurm_usage.model import Status


def controller_up() -> bool:
    if shutil.which("scontrol") is None:
        return False
    try:
        return subprocess.run(["scontrol", "ping"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(controller_up(), "no reachable Slurm controller")
class LiveSmokeTest(unittest.TestCase):
    def test_snapshot_and_gpu_reconciliation(self):
        collector = LiveCollector()
        for attempt in range(5):  # jobs start and end between the bulk calls
            snap = collector.snapshot()
            result = reconcile_gpus(snap)
            if result.explained:
                break
        self.assertIn(snap.capabilities["nodes"].status, (Status.OK, Status.EMPTY))
        self.assertIn(snap.capabilities["jobs"].status, (Status.OK, Status.EMPTY, Status.PARTIAL))
        gpus = snap.totals.gpus
        sys.stderr.write(
            "\n  live: %.2fs, %d nodes, %d job records; GPUs total=%s alloc=%s completing=%s "
            "free=%s unavailable=%s; jobs hold %s; difference=%s causes=%s; warnings=%s\n"
            % (snap.duration, len(snap.nodes or ()), len(snap.jobs or ()), gpus.total,
               gpus.alloc, gpus.completing, gpus.free, gpus.unavailable, result.jobs_alloc,
               result.difference, result.causes, [w.code for w in snap.warnings]))
        self.assertTrue(result.explained, "unexplained GPU difference: %r" % (result,))
        if None not in (gpus.total, gpus.alloc, gpus.free, gpus.unavailable, gpus.reserved,
                        gpus.shared, gpus.completing):
            self.assertEqual(gpus.total, gpus.alloc + gpus.completing + gpus.shared + gpus.free
                             + gpus.unavailable + gpus.reserved)
        for job in snap.jobs or ():
            if job.holds_resources:
                self.assertIsNotNone(job.user)


if __name__ == "__main__":
    unittest.main()
