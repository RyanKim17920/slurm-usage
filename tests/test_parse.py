import unittest

from slurm_usage import parse
from slurm_usage.parse import Frame


class UnitsTest(unittest.TestCase):
    def test_memory(self):
        self.assertEqual(parse.parse_memory("512000"), 512000 * 2**20)
        self.assertEqual(parse.parse_memory("16G"), 16 * 2**30)
        self.assertEqual(parse.parse_memory("1.5T"), int(1.5 * 2**40))
        self.assertEqual(parse.parse_memory("4000Mc"), 4000 * 2**20)
        self.assertEqual(parse.parse_memory("2048K", ""), 2048 * 1024)
        self.assertEqual(parse.parse_memory("123", ""), 123)
        self.assertEqual(parse.memory_basis("16Gn"), "node")
        for bad in ("", "N/A", "abc", None, "4294967294"):
            self.assertIsNone(parse.parse_memory(bad))

    def test_count(self):
        self.assertEqual(parse.parse_count("2K"), 2048)
        self.assertEqual(parse.parse_count("12"), 12)
        self.assertIsNone(parse.parse_count("x"))
        self.assertIsNone(parse.parse_count("9" * 5000))

    def test_int_sentinels(self):
        self.assertEqual(parse.parse_int("0"), 0)
        self.assertIsNone(parse.parse_int("4294967294"))
        self.assertIsNone(parse.parse_int("18446744073709551614"))
        self.assertIsNone(parse.parse_int(""))

    def test_durations(self):
        cases = {"1-02:03:04": 93784, "02:03:04": 7384, "03:04": 184, "5": 300,
                 "2-00:00:00": 172800, "1-2": 93600, "1-02:03": 93780,
                 "00:01.500": 1.5, "0:00": 0}
        for text, seconds in cases.items():
            self.assertEqual(parse.parse_duration(text), seconds, text)
        for text in ("UNLIMITED", "N/A", "", "INVALID", "Partition_Limit", None):
            self.assertIsNone(parse.parse_duration(text))
        self.assertTrue(parse.is_unlimited("UNLIMITED"))
        self.assertTrue(parse.is_unlimited("infinite"))
        self.assertFalse(parse.is_unlimited("N/A"))

    def test_garbage_duration(self):
        # sstat reports nonsense like this for .extern steps
        self.assertIsNone(parse.parse_duration("213503982334-14:25:51"))

    def test_time(self):
        self.assertEqual(parse.parse_time("1800000000"), 1800000000)
        for bad in ("Unknown", "None", "N/A", "0", "", "4294967294"):
            self.assertIsNone(parse.parse_time(bad))
        self.assertIsInstance(parse.parse_time("2024-01-02T03:04:05"), int)

    def test_exit_and_state(self):
        self.assertEqual(parse.parse_exit_code("1:9"), (1, 9))
        self.assertEqual(parse.parse_exit_code("0"), (0, None))
        self.assertEqual(parse.parse_state("CANCELLED by 2001"), "CANCELLED")
        self.assertEqual(parse.parse_state("completing+"), "COMPLETING")
        self.assertIsNone(parse.parse_state("(null)"))


class HostlistTest(unittest.TestCase):
    def test_expand(self):
        self.assertEqual(parse.expand_hostlist("gpu[01-03,07],cpu1"),
                         ["gpu01", "gpu02", "gpu03", "gpu07", "cpu1"])
        self.assertEqual(parse.expand_hostlist("r[1-2]n[0-1]"), ["r1n0", "r1n1", "r2n0", "r2n1"])
        self.assertEqual(parse.expand_hostlist("a[1-5:2]"), ["a1", "a3", "a5"])
        self.assertEqual(parse.expand_hostlist("x1,x1"), ["x1"])
        self.assertEqual(parse.expand_hostlist("(null)"), [])
        self.assertEqual(parse.expand_hostlist(""), [])

    def test_malformed_kept_literally(self):
        self.assertEqual(parse.expand_hostlist("bad[3-1]"), ["bad[3-1]"])
        self.assertEqual(parse.expand_hostlist("open[1-2"), ["open[1-2"])

    def test_limit(self):
        self.assertEqual(parse.expand_hostlist("n[0-99999999]", limit=1000), ["n[0-99999999]"])
        self.assertEqual(parse.expand_hostlist("a[1-100]b[1-100]", limit=1000),
                         ["a[1-100]b[1-100]"])

    def test_ranges(self):
        self.assertEqual(parse.parse_ranges("0-2,5"), [0, 1, 2, 5])
        self.assertEqual(parse.parse_ranges("7"), [7])
        self.assertIsNone(parse.parse_ranges("N/A"))
        self.assertIsNone(parse.parse_ranges("x"))
        self.assertEqual(parse.compress_ranges([5, 0, 1, 2, 9, 10]), "0-2,5,9-10")


class TresTest(unittest.TestCase):
    def test_tres(self):
        tres = parse.parse_tres("cpu=16,mem=220G,node=1,billing=16,gres/gpu=2,gres/gpu:a100=2")
        self.assertEqual(tres["cpu"], 16)
        self.assertEqual(tres["mem"], 220 * 2**30)
        self.assertEqual(parse.gpus_from_tres(tres), (2, {"a100": 2}))

    def test_typed_only_and_none(self):
        self.assertEqual(parse.gpus_from_tres(parse.parse_tres("gres/gpu:v100=3,cpu=1")),
                         (3, {"v100": 3}))
        self.assertEqual(parse.gpus_from_tres(parse.parse_tres("cpu=1")), (None, {}))

    def test_bare_mem_unit(self):
        self.assertEqual(parse.parse_tres("mem=1000")["mem"], 1000 * 2**20)
        self.assertEqual(parse.parse_usage_tres("mem=1000")["mem"], 1000)

    def test_usage_tres(self):
        usage = parse.parse_usage_tres(
            "cpu=00:01:30,energy=0,fs/disk=1024,gres/gpumem=2G,gres/gpuutil=180,mem=511560K")
        self.assertEqual(usage["cpu"], 90)
        self.assertEqual(usage["gres/gpuutil"], 180)
        self.assertEqual(usage["gres/gpumem"], 2 * 2**30)
        self.assertEqual(usage["mem"], 511560 * 1024)

    def test_unparsable_omitted(self):
        self.assertNotIn("cpu", parse.parse_tres("cpu=lots,node=1"))


class GresTest(unittest.TestCase):
    def counts(self, text):
        return parse.gpus_from_gres(parse.parse_gres(text))

    def test_typed_with_socket(self):
        self.assertEqual(self.counts("gpu:a100:8(S:0-1)"), (8, {"a100": 8}))

    def test_untyped(self):
        self.assertEqual(self.counts("gpu:4"), (4, {}))
        self.assertEqual(self.counts("gpu"), (1, {}))
        self.assertEqual(self.counts("(null)"), (0, {}))

    def test_multiple_types(self):
        self.assertEqual(self.counts("gpu:a100:2,gpu:v100:4"), (6, {"a100": 2, "v100": 4}))

    def test_mig(self):
        total, types = self.counts("gpu:1g.10gb:7(S:0),gpu:3g.40gb:2(S:1)")
        self.assertEqual(total, 9)
        self.assertEqual(types, {"1g.10gb": 7, "3g.40gb": 2})

    def test_no_double_count_of_generic_and_typed(self):
        self.assertEqual(self.counts("gpu:4,gpu:a100:4"), (4, {"a100": 4}))
        self.assertEqual(self.counts("gres/gpu=4,gres/gpu:a100=4"), (4, {"a100": 4}))

    def test_indices_with_commas(self):
        entries = parse.parse_gres("gpu:v100:4(IDX:0-1,3,5),gpu:l4:1(IDX:7)")
        self.assertEqual(parse.gpus_from_gres(entries), (5, {"v100": 4, "l4": 1}))
        self.assertEqual(parse.gpu_indices(entries), [0, 1, 3, 5, 7])

    def test_used_with_no_indices(self):
        entries = parse.parse_gres("gpu:a100:0(IDX:N/A)")
        self.assertEqual(parse.gpus_from_gres(entries), (0, {"a100": 0}))
        self.assertIsNone(parse.gpu_indices(entries))

    def test_shard_and_mps(self):
        entries = parse.parse_gres("gpu:2(IDX:0-1),shard:8(2/4,0/4),mps:200")
        self.assertEqual(parse.gpus_from_gres(entries), (2, {}))
        shard = [e for e in entries if e.name == "shard"][0]
        self.assertEqual(shard.shares, [(2, 4), (0, 4)])

    def test_no_consume_and_tres_spelling(self):
        self.assertEqual(self.counts("gpu:a100:no_consume:2"), (2, {"a100": 2}))
        self.assertEqual(self.counts("gres:gpu:3"), (3, {}))
        self.assertEqual(self.counts("gres/gpu:2"), (2, {}))

    def test_other_gres_ignored(self):
        self.assertEqual(self.counts("license:5,nic:2"), (0, {}))


class RecordsTest(unittest.TestCase):
    def test_key_values_spaces(self):
        kv = parse.key_values("NodeName=a OS=Linux 6.1 #1 SMP State=IDLE x=1")
        self.assertEqual(kv["OS"], "Linux 6.1 #1 SMP")
        self.assertEqual(kv["State"], "IDLE x=1")  # lower-case keys are not keys

    def test_key_values_first_wins(self):
        self.assertEqual(parse.key_values("A=1 B=2 A=3")["A"], "1")

    def test_key_needs_word_start(self):
        kv = parse.key_values("Name=xUserId=root A=1")
        self.assertEqual(kv["Name"], "xUserId=root")

    def test_frame_round_trip(self):
        frame = Frame.new()
        hostile = "a|b\nc %s fake" % "z00000s"
        text = frame.format(["1", hostile, "x"]) + "\n" + frame.format(["2", "ok", "y"])
        rows = list(frame.records(text.split("\n"), 3))
        self.assertEqual(rows, [["1", hostile, "x"], ["2", "ok", "y"]])

    def test_frame_wrong_width_and_cut(self):
        frame = Frame.new()
        lines = [frame.format(["1", "2"]), frame.begin + "1" + frame.separator + "2"]
        self.assertEqual(list(frame.records(lines, 3)), [None, None])

    def test_frames_differ(self):
        self.assertNotEqual(Frame.new(), Frame.new())


if __name__ == "__main__":
    unittest.main()


class BatchScriptBodyTest(unittest.TestCase):
    def test_shapes(self):
        dashes = "-" * 80
        self.assertIsNone(parse.batch_script_body(""))
        self.assertIsNone(parse.batch_script_body("\n\n"))
        self.assertEqual(parse.batch_script_body("Batch Script for 7\n%s\nNONE\n" % dashes), "")
        self.assertEqual(parse.batch_script_body("Batch Script for 7_3\n%s\n" % dashes), "")
        self.assertEqual(parse.batch_script_body(
            "Batch Script for 7\n%s\n#!/bin/sh\n  echo hi\n\n" % dashes), "#!/bin/sh\n  echo hi")
        self.assertEqual(parse.batch_script_body("#!/bin/sh\n"), "#!/bin/sh")
