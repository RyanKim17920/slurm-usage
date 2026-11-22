"""Parsing traps: framing, times, TRES, memory, states."""

import unittest

from tests.test_history_support import H, T0, FakeSlurm, job  # noqa: F401  (sets sys.path)

from slurm_usage.history import records
from slurm_usage.history.records import NONE, UNLIMITED, normalize, split_records


def as_row(**kw):
    row = job(**kw)
    row.pop("_steps")
    return {k: ("Unknown" if v is None else str(v)) for k, v in row.items()}


class FramingTest(unittest.TestCase):
    def test_hostile_name_cannot_forge_a_record(self):
        sep = records.new_delimiter()
        fields = ["JobIDRaw", "State", "JobName"]
        forged = "x\n999%sCOMPLETED%sevil" % (sep[:-1], sep[:-1])  # without the token
        lines = ["1%sCOMPLETED%s%s" % (sep, sep, forged), "2%sFAILED%sok" % (sep, sep)]
        text = "\n".join(lines).split("\n")
        out = list(split_records(text, sep, len(fields)))
        self.assertEqual([r[0] for r in out], ["1", "2"])
        self.assertEqual(out[0][2], forged)

    def test_delimiters_are_unpredictable(self):
        self.assertNotEqual(records.new_delimiter(), records.new_delimiter())

    def test_short_record_is_reported(self):
        sep = records.new_delimiter()
        out = list(split_records(["1%sCOMPLETED" % sep], sep, 3))
        self.assertEqual(out, [None])

    def test_bulk_queries_refuse_free_text_fields(self):
        from slurm_usage.history.commands import bulk_sacct
        for field in ("SubmitLine", "WorkDir", "Comment"):
            with self.assertRaises(ValueError):
                bulk_sacct(["JobIDRaw", field], "|")


class NormalizeTest(unittest.TestCase):
    def test_exact_gpu_key_wins_over_typed(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H,
                             alloc="cpu=4,gres/gpu=2,gres/gpu:a100=2,node=1"))
        self.assertEqual(r.gpus, 2)

    def test_typed_gpu_keys_summed_without_generic(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H,
                             alloc="cpu=4,gres/gpu:a100=2,gres/gpu:v100=1,node=1"))
        self.assertEqual(r.gpus, 3)

    def test_allocation_without_gpus_is_zero_not_unknown(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, gpus=0))
        self.assertEqual(r.gpus, 0)

    def test_memory_from_request_when_not_allocated(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, mem="16G"))
        self.assertEqual(r.mem, 16 * 1024 ** 3)

    def test_old_per_cpu_and_per_node_memory(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, cpus=4, req="cpu=4,node=1",
                             req_mem="1000Mc"))
        self.assertEqual(r.mem, 4 * 1000 * 1024 ** 2)
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, req="cpu=4,node=1",
                             req_mem="2Gn", NNodes="3"))
        self.assertEqual(r.mem, 3 * 2 * 1024 ** 3)

    def test_unknown_memory_is_none(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, req="cpu=4,node=1", req_mem="0"))
        self.assertIsNone(r.mem)

    def test_pending_has_no_allocation(self):
        r = normalize(as_row(job_id=1, state="PENDING", start=None, end=None))
        self.assertIsNone(r.start)
        self.assertIsNone(r.cpus)
        self.assertIsNone(r.gpus)

    def test_pending_start_estimate_is_ignored(self):
        r = normalize(as_row(job_id=1, state="PENDING", start=T0 + 9 * H, end=None))
        self.assertIsNone(r.start)

    def test_cancelled_while_pending_has_no_allocation(self):
        r = normalize(as_row(job_id=1, state="CANCELLED by 5", start=T0, end=T0, alloc="",
                             AllocCPUS="0"))
        self.assertIsNone(r.start)

    def test_suspended_and_end_before_start(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + 4 * H, suspended=H))
        self.assertEqual(r.suspended, H)
        r = normalize(as_row(job_id=1, start=T0 + H, end=T0))
        self.assertEqual(r.end, r.start)

    def test_state_and_limits(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, state="CANCELLED by 123",
                             limit="UNLIMITED"))
        self.assertEqual(r.roles.state, "CANCELLED")
        self.assertEqual(r.limit, UNLIMITED)
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, limit="Partition_Limit"))
        self.assertIsNone(r.limit)

    def test_missing_roles_and_steps(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, user="", account=""))
        self.assertEqual(r.roles.user, "uid:5001")
        self.assertEqual(r.roles.account, NONE)
        self.assertIsNone(normalize(as_row(job_id="1.batch", start=T0, end=T0 + H)))

    def test_hostile_role_values_are_cleaned(self):
        r = normalize(as_row(job_id=1, start=T0, end=T0 + H, user="ev\x1b[2Jil",
                             name="a\x1b]0;x\x07b"))
        self.assertNotIn("\x1b", r.roles.user)
        self.assertNotEqual(r.roles.user, "evil")  # cannot merge with a real user
        self.assertEqual(r.name, "ab")

    def test_iso_times_still_parse(self):
        row = as_row(job_id=1, start=T0, end=T0 + H)
        import time
        row["Start"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(T0))
        self.assertEqual(normalize(row).start, T0)

    def test_running_suspended_job_counts_current_suspension(self):
        row = as_row(job_id=1, start=T0, end=None, state="SUSPENDED")
        row["ElapsedRaw"] = str(H)
        row["Suspended"] = "00:00:00"
        r = normalize(row, seen_at=T0 + 3 * H)
        self.assertEqual(r.suspended, 2 * H)


if __name__ == "__main__":
    unittest.main()
