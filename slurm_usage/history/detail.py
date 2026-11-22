"""One job, asked from Slurm at the moment someone looks at it. Nothing is cached.

Works for any job. Slurm decides what the invoker may read; a refusal is
reported as ``denied``. Job environments are never requested.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from ..model import Status, Step
from ..parse import (SCRIPT_NOT_STORED, SCRIPT_NOT_VISIBLE, batch_script_body, expand_hostlist,
                     optional_text, parse_duration, parse_exit_code, parse_int,
                     parse_memory, parse_state, parse_time, parse_tres, parse_usage_tres)
from ..sanitize import clean, clean_multiline
from .commands import stream_lines
from .efficiency import AttemptUsage
from .environment import Environment
from .errors import HistoryError
from .listing import JobRow, make_row
from .paths import LogPath, expand_path
from .records import new_delimiter, normalize, outcome_of, read_rows
from .select import unresolved
from .store import FLAG_NONE, Info, Row

DETAIL_FIELDS = (
    "DBIndex", "JobIDRaw", "Cluster", "JobID", "Submit", "Eligible", "Start", "End", "Suspended",
    "ElapsedRaw", "TimelimitRaw", "State", "ExitCode", "Restarts", "UID", "User", "Account",
    "QOS", "Partition", "NNodes", "NTasks", "AllocCPUS", "AllocTRES", "ReqTRES", "ReqMem",
    "Reason", "TotalCPU", "AveCPU", "MaxRSS", "AveRSS", "TRESUsageInTot", "TRESUsageInMax",
    "TRESUsageInAve", "NodeList",
    # user-controlled
    "JobName", "WorkDir", "StdOut", "StdErr",
)
DETAIL_TIMEOUT = 60.0
SCRIPT_LIMIT = 4 * 1024 * 1024

_JOB_ID = re.compile(r"[0-9]{1,18}(?:_[0-9]{1,10}|\+[0-9]{1,5})?")


@dataclass
class AttemptDetail:
    job: JobRow
    exit_code: Optional[int] = None
    exit_signal: Optional[int] = None
    reason: Optional[str] = None
    workdir: Optional[str] = None  # sanitized, for display
    raw_workdir: Optional[str] = field(default=None, repr=False)  # for os.stat, never display
    stdout: Optional[LogPath] = None
    stderr: Optional[LogPath] = None
    steps: List[Step] = field(default_factory=list)


@dataclass
class JobDetail:
    job_id: str
    status: str
    message: str = ""
    attempts: List[AttemptDetail] = field(default_factory=list)  # oldest first


class Text(NamedTuple):
    """status: ok, empty, not_stored, denied, unavailable, timeout, partial or error."""

    status: str
    text: str = ""
    message: str = ""

    def __repr__(self) -> str:  # so scripts and submit lines do not end up in logs
        return "Text(status=%r, %d characters, message=%r)" % (self.status, len(self.text),
                                                               self.message)


def check_job_id(job_id: object) -> str:
    text = str(job_id)
    if not _JOB_ID.fullmatch(text):
        raise ValueError("not a job id")
    return text


def _make_step(row: Dict[str, str]) -> Step:
    job, _, step = row.get("JobIDRaw", "").partition(".")
    total = parse_usage_tres(row.get("TRESUsageInTot"))
    ave = parse_usage_tres(row.get("TRESUsageInAve"))
    top = parse_usage_tres(row.get("TRESUsageInMax"))
    code, signal = parse_exit_code(row.get("ExitCode"))
    sampled = bool(total.get("mem"))

    def number(source: Dict[str, float], key: str) -> Optional[int]:
        value = source.get(key)
        return int(value) if value else None

    def util(source: Dict[str, float]) -> Optional[float]:
        value = source.get("gres/gpuutil")
        return value if value or (value is not None and sampled) else None

    return Step(
        job_id=clean(job), step_id=clean(step), name=clean(row.get("JobName")) or None,
        state=clean(parse_state(row.get("State"))) or None,
        start=parse_time(row.get("Start")), end=parse_time(row.get("End")),
        elapsed=parse_int(row.get("ElapsedRaw")), exit_code=code, exit_signal=signal,
        ntasks=parse_int(row.get("NTasks")),
        nodes=[clean(node) for node in expand_hostlist(row.get("NodeList"))],
        ave_cpu=parse_duration(row.get("AveCPU")) or None,
        total_cpu=parse_duration(row.get("TotalCPU")) or None,
        ave_rss=parse_memory(row.get("AveRSS"), "") or None,
        max_rss=parse_memory(row.get("MaxRSS"), "") or None,
        gpu_util_ave=util(ave), gpu_util_max=util(top), gpu_util_total=util(total),
        gpu_mem_max=number(top, "gres/gpumem"), gpu_mem_total=number(total, "gres/gpumem"),
        tres_alloc=parse_tres(row.get("AllocTRES")), tres_usage_ave=ave, tres_usage_max=top,
        tres_usage_total=total)


def fetch_detail(runner: Any, env: Environment, job_id: object, as_of: int,
                 flags: Optional[Dict[str, int]] = None,
                 cancelled: Callable[[], bool] = lambda: False) -> JobDetail:
    """Every attempt and step of one job, by raw id or display id."""
    asked = check_job_id(job_id)
    fields = env.supported(DETAIL_FIELDS)
    separator = new_delimiter()
    argv = ["sacct", "-a", "-D", "-n", "-P", "--delimiter=" + separator, "-j", asked,
            "-o", ",".join(fields)]
    detail = JobDetail(asked, Status.EMPTY)
    by_id = "JobIDRaw" if asked.isdigit() else "JobID"
    building = []  # (record, its row, its steps, usage)
    current, lines = None, stream_lines(runner, argv, cancelled, DETAIL_TIMEOUT)
    try:
        for row in read_rows(lines, separator, fields):
            if row is None:
                current = None
                continue
            raw = row.get("JobIDRaw", "")
            if "." in raw:
                if current is not None and raw.partition(".")[0] == str(current[0].job):
                    current[2].append(_make_step(row))
                    current[3].step(row.get("TotalCPU"), row.get("TRESUsageInTot"))
                continue
            current = None
            cluster = row.get("Cluster")
            if cluster and env.cluster and cluster.lower() != env.cluster.lower():
                continue
            # an array's or het job's id also returns its other members
            if row.get(by_id) != asked:
                continue
            record = normalize(row, as_of)
            if record is not None:
                current = (record, row, [], AttemptUsage(row.get("TotalCPU")))
                building.append(current)
    except HistoryError as error:
        detail.status, detail.message = error.status, error.message
    finally:
        lines.close()

    building.sort(key=lambda item: (item[0].submit or 0, item[0].start or 0))
    for record, row, steps, usage in building:
        used = usage.result() if record.end is not None else (None, None, None)
        flag = (flags or {}).get(record.key, FLAG_NONE)
        problem = unresolved(record.roles.state, record.end, flag) if record.start else None
        stored = Row(record.key, record.job, 0, 0, record.submit, record.eligible, record.start,
                     record.end, record.suspended, record.cpus, record.gpus, record.mem,
                     record.limit, flag, *used, True)
        real_stop = None
        if record.start is not None and not problem:
            real_stop = record.end if record.end is not None else as_of
        if problem or (record.start is None and record.end is None):
            outcome = "other"
        else:
            outcome = outcome_of(record.roles.state, record.end is not None)
        job = make_row(stored, record.roles, Info(record.label, record.name, record.restarts),
                       record.nodes, outcome, problem, real_stop)
        if job.elapsed is not None:
            for name, amount in (("gpu_seconds", job.gpus), ("cpu_seconds", job.cpus),
                                 ("mem_byte_seconds", job.mem)):
                setattr(job, name, None if amount is None else amount * job.elapsed)
        code, signal = parse_exit_code(row.get("ExitCode"))
        item = AttemptDetail(job, exit_code=code, exit_signal=signal, steps=steps,
                             reason=clean(optional_text(row.get("Reason"))) or None,
                             workdir=clean(row.get("WorkDir")) or None,
                             raw_workdir=row.get("WorkDir") if optional_text(row.get("WorkDir")) else None)
        context = (str(record.job), row.get("JobID", ""), row.get("User", ""),
                   row.get("JobName", ""), record.nodes, row.get("WorkDir", ""))
        item.stdout = expand_path(row.get("StdOut"), *context)
        if optional_text(row.get("StdErr")):
            item.stderr = expand_path(row.get("StdErr"), *context)
        elif item.stdout is not None and "StdErr" in row:
            item.stderr = item.stdout._replace(merged=True)
        detail.attempts.append(item)

    if detail.status == Status.EMPTY:
        if detail.attempts:
            detail.status = Status.OK
        else:
            detail.message = "no such job in accounting, or Slurm does not show it to you"
    return detail


def fetch_submit_line(runner: Any, job_id: object, key: Optional[str] = None) -> Text:
    """Call only when the viewer asked for this job's submit line. Never cached."""
    job = check_job_id(job_id)
    if not job.isdigit():
        raise ValueError("use the numeric job id")
    fields = ("DBIndex", "JobIDRaw", "Submit", "SubmitLine")
    separator = new_delimiter()
    result = runner.run(["sacct", "-a", "-D", "-X", "-n", "-P", "--delimiter=" + separator,
                         "-j", job, "-o", ",".join(fields)], timeout=DETAIL_TIMEOUT)
    if not result.ok:
        return Text(result.status, message=result.reason)
    chosen = None
    for row in read_rows(result.lines(), separator, fields):
        if not row or row["JobIDRaw"] != job or (key and row["DBIndex"] != key):
            continue
        if chosen is None or (parse_int(row["Submit"]) or 0) >= (parse_int(chosen["Submit"]) or 0):
            chosen = row
    if chosen is None:
        if result.status == Status.PARTIAL:
            return Text(result.status, message=result.reason)
        return Text(Status.EMPTY, message="no such job, or Slurm does not show it to you")
    if not chosen["SubmitLine"].strip():
        return Text(Status.EMPTY, message="no submit line recorded")
    return Text(Status.OK, clean_multiline(chosen["SubmitLine"]))


def fetch_batch_script(runner: Any, job_id: object) -> Text:
    """Call only when the viewer asked for this job's script. Never cached."""
    job = check_job_id(job_id)
    if not job.isdigit():
        raise ValueError("use the numeric job id")
    result = runner.batch_script(job)
    if not result.ok:
        return Text(result.status, message=result.reason)
    body = batch_script_body(result.stdout)
    if body is None:
        if result.status == Status.PARTIAL:
            # headers on stdout, the refusal on stderr
            from ..runner import classify
            status = classify(result.returncode, False, result.stderr)
            return Text(status if status != Status.EMPTY else Status.DENIED,
                        message=result.reason)
        return Text(Status.DENIED, message=result.reason or SCRIPT_NOT_VISIBLE)
    if not body.strip():
        if result.status == Status.PARTIAL:
            from ..runner import classify
            status = classify(result.returncode, False, result.stderr)
            if status != Status.EMPTY:
                return Text(status, message=result.reason)
        return Text(Status.NOT_STORED, message=SCRIPT_NOT_STORED)
    if len(body) > SCRIPT_LIMIT:
        return Text(Status.PARTIAL, clean_multiline(body[:SCRIPT_LIMIT]), "script truncated")
    return Text(result.status, clean_multiline(body), result.reason)
