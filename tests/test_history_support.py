"""A fake Slurm for the history tests. Invented data only; no tests in this file."""

import os
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slurm_usage.model import Status  # noqa: E402
from slurm_usage.runner import Result, check  # noqa: E402

T0 = 1700000000 - 1700000000 % 86400  # a UTC midnight
H = 3600

CONFIG = """ClusterName = testcluster
AccountingStorageType = accounting_storage/slurmdbd
AccountingStorageHost = dbd.example
AccountingStorageTRES = cpu,mem,energy,node,billing,gres/gpu
PrivateData = (null)
SlurmUser = slurm(64030)
"""
ALL_FIELDS = """Account AllocCPUS AllocTRES AveCPU AveRSS Cluster DBIndex Elapsed ElapsedRaw
Eligible End ExitCode JobID JobIDRaw JobName MaxRSS NNodes NodeList NTasks Partition QOS Reason
ReqMem ReqTRES Restarts Start State StdErr StdOut Submit SubmitLine Suspended TimelimitRaw
TotalCPU TRESUsageInAve TRESUsageInMax TRESUsageInTot UID User WorkDir"""


def stamp_to_epoch(text: str, now: Optional[int] = None) -> Optional[int]:
    if text == "now":
        return None
    if text.startswith("now-") and text.endswith("days"):
        return (now or 0) - int(text[4:-4]) * 86400
    if text.startswith("now-") and text.endswith("hours"):
        return (now or 0) - int(text[4:-5]) * 3600
    return int(time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S")))


def job(job_id, start=None, end=None, state="COMPLETED", user="alice", uid=5001,
        account="physics", qos="normal", partition="gpu", gpus=1, cpus=4, mem="16G",
        nodes="g01", submit=None, eligible=None, suspended=0, limit=60, name="train",
        label=None, index=None, alloc=None, req=None, req_mem=None, restarts=0, steps=(),
        **extra) -> Dict[str, object]:
    """One allocation row. Times are epoch ints or None (printed as Unknown)."""
    submit = (start if start is not None else T0) if submit is None else submit
    alloc_tres = alloc if alloc is not None else (
        "" if start is None else "billing=%d,cpu=%d,%snode=1" % (
            cpus, cpus, "gres/gpu=%d," % gpus if gpus else ""))
    row = dict(
        DBIndex=str(index if index is not None else abs(hash((job_id, submit, start))) % 10**12),
        JobIDRaw=str(job_id), Cluster="testcluster", JobID=label or str(job_id),
        Submit=submit, Eligible=submit if eligible is None else eligible, Start=start, End=end,
        Suspended="%02d:%02d:%02d" % (suspended // 3600, suspended % 3600 // 60, suspended % 60),
        ElapsedRaw=str(0 if start is None else max(0, (end if end is not None else start) - start
                                                    - suspended)),
        State=state, AllocTRES=alloc_tres,
        ReqTRES=req if req is not None else "cpu=%d,mem=%s,node=1%s" % (
            cpus, mem, ",gres/gpu=%d" % gpus if gpus else ""),
        ReqMem=req_mem if req_mem is not None else mem, NNodes="1", AllocCPUS=str(cpus),
        TimelimitRaw=str(limit), Restarts=str(restarts), UID=str(uid), User=user,
        Account=account, QOS=qos, Partition=partition,
        NodeList=nodes if start is not None else "None assigned", JobName=name,
        ExitCode="0:0", NTasks="", Reason="None", WorkDir="/work/" + user,
        StdOut="/work/%s/slurm-%%j.out" % user, StdErr="", SubmitLine="sbatch run.sh",
        TotalCPU="00:00:00", TRESUsageInTot="", TRESUsageInMax="", TRESUsageInAve="",
        AveCPU="", AveRSS="", MaxRSS="")
    row.update({k: v for k, v in extra.items()})
    row["_steps"] = list(steps)
    return row


def step(suffix, **fields) -> Dict[str, object]:
    return dict(suffix=suffix, **fields)


class FakeStream:
    def __init__(self, lines: List[str], status: str, reason: str = "",
                 argv: Optional[List[str]] = None, on_line=None) -> None:
        self._lines, self._on_line = lines, on_line
        self.result = None  # type: Optional[Result]
        self._final = Result(list(argv or []), status, "", reason if status != Status.OK else "",
                             0 if status in (Status.OK, Status.EMPTY) else 1, reason)
        self.closed_early = False

    def __iter__(self):
        if self._final.status not in Status.USABLE:
            self.result = self._final
            return
        for at, line in enumerate(self._lines):
            if self.result is not None:
                return
            if self._on_line:
                self._on_line(at)
            yield line
        self.result = self.result or self._final

    def close(self) -> None:
        if self.result is None:
            self.closed_early = True
            self.result = Result(self._final.argv, Status.CANCELLED, "", "", None, "stopped")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeSlurm:
    """Answers the queries the history layer makes, from a list of job rows.

    Every argv is checked against the real read-only policy first.
    """

    def __init__(self, jobs=(), now: int = T0 + 30 * 86400, live=None, config: str = CONFIG,
                 helpformat: str = ALL_FIELDS, cluster_tres: str = "cpu=64,gres/gpu=16",
                 nodes: Optional[List[str]] = None, admin: str = "None",
                 coordinator: str = "") -> None:
        self.jobs = list(jobs)
        self.now = now
        self.live = live  # job ids squeue shows; None = squeue fails
        self.config, self.helpformat, self.cluster_tres = config, helpformat, cluster_tres
        self.nodes = nodes if nodes is not None else [
            "NodeName=g01 CPUTot=32 CfgTRES=cpu=32,mem=256G,gres/gpu=8 Gres=gpu:8",
            "NodeName=g02 CPUTot=32 CfgTRES=cpu=32,mem=256G,gres/gpu=8 Gres=gpu:8"]
        self.admin, self.coordinator = admin, coordinator
        self.calls = []  # type: List[List[str]]
        self.fail = {}  # type: Dict[str, str]  # argv substring -> status
        self.scripts = {}  # type: Dict[str, Result]
        self.on_line = None

    # -- the runner interface -------------------------------------------------
    def run(self, argv, timeout=None, max_bytes=None) -> Result:
        with self.stream(argv) as stream:
            lines = list(stream)
        result = stream.result
        if lines:
            result.stdout = "\n".join(lines) + "\n"
        return result

    def stream(self, argv, timeout=None, max_bytes=None) -> FakeStream:
        check(argv)
        argv = list(argv)
        self.calls.append(argv)
        joined = " ".join(argv)
        for needle, status in self.fail.items():
            if needle in joined:
                return FakeStream([], status, "simulated %s" % status, argv)
        lines = self._answer(argv)
        if lines is None:
            return FakeStream([], Status.ERROR, "unsupported by the fake", argv)
        return FakeStream(lines, Status.OK if lines else Status.EMPTY, "", argv, self.on_line)

    def batch_script(self, job_id: str) -> Result:
        argv = ["sacct", "-j", str(job_id), "--batch-script"]
        check(argv)
        self.calls.append(argv)
        return self.scripts.get(str(job_id), Result(argv, Status.EMPTY, "", "", 0, ""))

    # -- answers ------------------------------------------------------------
    def _answer(self, argv: List[str]) -> Optional[List[str]]:
        program = argv[0]
        if program == "scontrol" and "config" in argv:
            return self.config.splitlines()
        if program == "scontrol" and "node" in argv:
            return list(self.nodes)
        if program == "squeue":
            if self.live is None:
                return None
            return [str(j) for j in self.live]
        if program == "sacctmgr":
            if "user" in argv:
                return ["%s|%s|%s" % (os.environ.get("USER", "x"), self.admin, self.coordinator)]
            if "cluster" in argv:
                return ["testcluster|%s" % self.cluster_tres]
            return []
        if program == "sacct":
            if "--helpformat" in argv:
                return self.helpformat.split("\n")
            return self._sacct(argv)
        return None

    def _sacct(self, argv: List[str]) -> List[str]:
        options = {}
        index = 1
        while index < len(argv):
            token = argv[index]
            if token.startswith("--delimiter="):
                options["delimiter"] = token.split("=", 1)[1]
            elif token in ("-S", "-E", "-j", "-o", "-u", "-A"):
                options[token] = argv[index + 1]
                index += 1
            else:
                options.setdefault("flags", set()).add(token)
            index += 1
        fields = options["-o"].split(",")
        separator = options.get("delimiter", "|")
        steps = "-X" not in options.get("flags", set())
        low = stamp_to_epoch(options["-S"], self.now) if "-S" in options else None
        high = stamp_to_epoch(options["-E"], self.now) if "-E" in options else None
        wanted = set(options["-j"].split(",")) if "-j" in options else None
        out = []
        for row in self.jobs:
            if wanted is not None and row["JobIDRaw"] not in wanted and row["JobID"] not in wanted:
                continue
            if "-u" in options and row["UID"] != options["-u"]:
                continue
            if "-A" in options and row["Account"] != options["-A"]:
                continue
            if "UID" in fields and len(fields) == 1 and "-S" in options:
                pass
            begin = row["Start"] if row["Start"] is not None else row["Eligible"]
            if high is not None and begin is not None and begin > high:
                continue
            if low is not None and row["End"] is not None and row["End"] < low:
                continue
            out.append(self._format(row, fields, separator))
            if steps:
                for item in row["_steps"]:
                    values = dict(row)
                    values.update(JobIDRaw="%s.%s" % (row["JobIDRaw"], item["suffix"]),
                                  JobID="%s.%s" % (row["JobID"], item["suffix"]))
                    values.update({k: v for k, v in item.items() if k != "suffix"})
                    out.append(self._format(values, fields, separator))
        return "\n".join(out).split("\n") if out else []

    @staticmethod
    def _format(row, fields, separator) -> str:
        values = []
        for name in fields:
            value = row.get(name, "")
            if value is None:
                value = "Unknown"
            values.append(str(value))
        return separator.join(values)

    def sacct_calls(self, *needles: str) -> List[List[str]]:
        return [c for c in self.calls if c[0] == "sacct" and all(n in c for n in needles)]


@contextmanager
def timezone(name: str):
    old = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = old
        time.tzset()


def make_history(fake: FakeSlurm, persist: bool = False, cache_dir: Optional[str] = None):
    from slurm_usage.history import History
    return History(fake, cache_dir=cache_dir, persist=persist, clock=lambda: fake.now)


def temp_dir(case) -> str:
    directory = tempfile.mkdtemp(prefix="history-test-")
    case.addCleanup(__import__("shutil").rmtree, directory, True)
    return directory
