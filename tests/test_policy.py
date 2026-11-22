import unittest

from slurm_usage.runner import ReadOnlyViolation, check

ALLOWED = [
    ["squeue", "-h", "-a", "-r", "-t", "all", "-o", "%i|%j"],
    ["squeue", "--me", "-h"],
    ["squeue", "-h", "-a", "-j", "12", "-O", "JobID:|,STDOUT:|"],
    ["sinfo", "-N", "-h", "-o", "%N %T"],
    ["sacct", "-a", "-X", "-n", "-P", "-S", "now-1days", "-E", "now", "-o", "JobID,State"],
    ["sacct", "-nPXa", "-Snow-7days", "--format=JobID"],
    ["sacct", "--helpformat"],
    ["sacct", "-j", "12", "--batch-script"],
    ["sacct", "-X", "-n", "-P", "-j", "12", "-o", "JobIDRaw,SubmitLine"],
    ["sreport", "-n", "-P", "cluster", "Utilization", "start=now-7days"],
    ["sshare", "-a", "-P", "-n"],
    ["sprio", "-w", "-h"],
    ["sstat", "-a", "-n", "-P", "-j", "12", "-o", "JobID,AveCPU"],
    ["scontrol", "-a", "-d", "-o", "show", "job"],
    ["scontrol", "show", "assoc_mgr"],
    ["scontrol", "ping"],
    ["scontrol", "show", "hostnames", "gpu[01-02]"],
    ["sacctmgr", "-n", "-P", "show", "assoc", "format=account,user"],
    ["sacctmgr", "-n", "-P", "list", "qos"],
    ["sacctmgr", "-n", "-P", "show", "user", "alice", "withcoord"],
    ["sinfo", "--version"],
]

REFUSED = [
    ["scancel", "12"],
    ["sbatch", "run.sh"],
    ["bash", "-c", "squeue"],
    "squeue -h",
    [],
    ["scontrol", "update", "JobId=12", "Priority=1"],
    ["scontrol", "hold", "12"],
    ["scontrol", "write", "batch_script", "12"],
    ["scontrol", "-dd", "show", "job", "12"],
    ["scontrol", "-d", "--details", "show", "job", "12"],
    ["scontrol", "-u", "2002", "show", "job"],
    ["scontrol", "show", "job", "-u", "2002"],
    ["scontrol", "show", "hostnames", "/etc/hosts"],
    ["scontrol", "show", "hostnames"],
    ["scontrol", "reconfigure"],
    ["scontrol"],
    ["sacct", "--env-vars", "-j", "12"],
    ["sacct", "--env", "-j", "12"],
    ["sacct", "-j", "12", "-B"],
    ["sacct", "-B", "-j", "12"],
    ["sacct", "--batch-script", "-a"],
    ["sacct", "-a", "--batch-script"],
    ["sacct", "-j", "12,13", "--batch-script"],
    ["sacct", "-a", "-o", "JobID,SubmitLine"],
    ["sacct", "-a", "-o", "ALL"],
    ["sacct", "-j", "12,13", "-o", "submitline"],
    ["sacct", "--json"],
    ["sacct", "-f", "/var/log/slurm/accounting"],
    ["sacct", "-j", "-B"],
    ["sacctmgr", "add", "user", "bob", "account=acct-a"],
    ["sacctmgr", "-i", "show", "qos"],
    ["sacctmgr", "modify", "qos", "normal", "set", "priority=1"],
    ["sacctmgr", "show", "runawayjobs"],
    ["sacctmgr", "dump", "testcluster"],
    ["sacctmgr", "show"],
    ["sreport", "--json"],
    ["sstat", "-j", "12", "--env"],
    ["squeue", "--json"],
    ["squeue", "-h", "12"],
    ["squeue", "-o"],
    ["sinfo", "-i", "5"],
    ["squeue", "-h\0x"],
]


class PolicyTest(unittest.TestCase):
    def test_allowed(self):
        for argv in ALLOWED:
            with self.subTest(argv=argv):
                check(argv)

    def test_refused(self):
        for argv in REFUSED:
            with self.subTest(argv=argv):
                with self.assertRaises(ReadOnlyViolation):
                    check(argv)

    def test_live_queries_are_allowed(self):
        from slurm_usage.live import queries
        from slurm_usage.parse import Frame
        frame = Frame.new()
        for argv in (queries.NODES, queries.JOB_DETAIL, queries.PARTITIONS,
                     queries.RESERVATIONS, queries.ASSOC_MGR, queries.PING, queries.CONFIG,
                     queries.COORDINATORS, queries.ACCOUNTING_CAPACITY, queries.QOS,
                     queries.queue(frame), queries.associations("c1"), queries.user("alice"),
                     queries.steps("12"), queries.log_paths("12", frame),
                     queries.submit_line("12", "zdelimz")):
            check(argv)

    def test_job_id_injection_refused(self):
        from slurm_usage.live import queries
        with self.assertRaises(ReadOnlyViolation):
            check(queries.submit_line("12,13", "z"))


if __name__ == "__main__":
    unittest.main()
