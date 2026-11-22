"""Queries made only for a job the viewer opened: one job, one call.

Nothing here checks ownership. Slurm answers or refuses, and a refusal is
returned as ``denied``. Scripts and submit lines are sanitized and never cached.
"""

import re
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .. import parse
from ..model import Job, Status, Step
from ..sanitize import clean, clean_multiline
from . import queries

_PATTERN = re.compile(r"%([0-9]{0,2})([%AaJjNbnstuxX])")


@dataclass
class Fetched:
    status: str
    text: str = ""  # sanitized
    detail: str = ""


@dataclass
class StepReport:
    status: str
    steps: List[Step] = field(default_factory=list)
    detail: str = ""


@dataclass
class LogPaths:
    status: str
    stdout: Optional[str] = None  # sanitized, for display
    stderr: Optional[str] = None
    raw_stdout: Optional[str] = field(default=None, repr=False)  # for open(), never display
    raw_stderr: Optional[str] = field(default=None, repr=False)
    unresolved: List[str] = field(default_factory=list)
    detail: str = ""


def expand_pattern(template: str, job: Job) -> Tuple[str, List[str]]:
    """sbatch's filename patterns, as seen from the batch script."""
    if "\\" in template:  # a backslash turns expansion off
        return template.replace("\\", ""), []
    task = job.array_task if job.array_task and job.array_task.isdigit() else None
    host = (job.batch_host or (job.nodes[0] if job.nodes else "")).split(".")[0] or None
    values = {
        "%": "%", "j": job.job_id, "A": job.array_job_id or job.job_id,
        "a": task if job.array_job_id else str(2**32 - 2),
        "b": str(int(task) % 10) if task else None,
        "x": job.raw_name, "u": job.user, "N": host, "n": "0", "t": "0", "s": "batch",
        "J": None, "X": None,
    }  # type: Dict[str, Optional[str]]
    missing = []  # type: List[str]

    def replace(match: "re.Match") -> str:
        width, code = match.groups()
        value = values[code]
        if value is None:
            missing.append("%" + code)
            return match.group(0)
        return value.zfill(min(int(width), 10)) if width and value.isdigit() else value

    return _PATTERN.sub(replace, template), missing


def fetch_log_paths(runner, job: Job) -> LogPaths:
    if not job.job_id.isdigit():
        return LogPaths(Status.EMPTY, detail="not a numeric job id")
    frame = parse.Frame.new()
    result = runner.run(queries.log_paths(job.job_id, frame))
    if not result.ok:
        return LogPaths(result.status, detail=result.reason)
    for record in result.stdout.split(frame.end):
        values = record.lstrip("\n").split(frame.separator)
        if len(values) == 3 and values[0].strip() == job.job_id:
            break
    else:
        if result.stdout and frame.end not in result.stdout:
            return LogPaths(Status.UNAVAILABLE, detail="this squeue cannot report log paths")
        return LogPaths(Status.EMPTY,
                        detail=result.reason or "job is no longer known to the controller")
    paths = LogPaths(Status.OK)
    out, err = [parse.optional_text(value) for value in values[1:]]
    err = err or out  # no separate stderr means it follows stdout
    for attr, template in (("stdout", out), ("stderr", err)):
        if not template:
            continue
        path, missing = expand_pattern(template, job)
        if not path.startswith("/") and job.raw_workdir:
            path = job.raw_workdir.rstrip("/") + "/" + path
        paths.unresolved.extend(missing)
        setattr(paths, "raw_" + attr, path)
        setattr(paths, attr, clean(path))
    if not out:
        paths.status, paths.detail = Status.EMPTY, "this job has no log file"
    elif paths.unresolved:
        paths.status, paths.detail = Status.PARTIAL, "some filename patterns are not known yet"
    return paths


def parse_steps(text: str, job: Job) -> List[Step]:
    steps = []
    ceiling = None
    if job.elapsed is not None and job.cpus_alloc:
        ceiling = job.elapsed * job.cpus_alloc * 1.5 + 300  # more CPU than this is impossible
    for line in text.split("\n"):
        row = line.split("|")
        if len(row) != len(queries.STEP_FIELDS):
            continue
        ident, ntasks, ave_cpu, ave_rss, max_rss, ave, most, total, nodes = row
        base, _, step_id = ident.partition(".")
        if not step_id or base not in (job.job_id, job.display_id):
            continue
        step = Step(job.job_id, clean(step_id), ntasks=parse.parse_int(ntasks),
                    nodes=[clean(node) for node in parse.expand_hostlist(nodes)],
                    ave_rss=parse.parse_memory(ave_rss, ""),
                    max_rss=parse.parse_memory(max_rss, ""))
        step.ave_cpu = parse.parse_duration(ave_cpu) if ":" in ave_cpu else None
        step.tres_usage_ave = parse.parse_usage_tres(ave)
        step.tres_usage_max = parse.parse_usage_tres(most)
        step.tres_usage_total = parse.parse_usage_tres(total)
        usages = (step.tres_usage_ave, step.tres_usage_max, step.tres_usage_total)
        if step.step_id == "extern":
            step.ave_cpu = None
            for usage in usages:
                usage.pop("cpu", None)
            step.notes.append("CPU time of the extern step is not meaningful")
        elif (":" in ave_cpu and step.ave_cpu is None) or (
                ceiling is not None and step.ave_cpu is not None and step.ave_cpu > ceiling):
            step.ave_cpu = None
            step.notes.append("impossible CPU time reported")
        step.total_cpu = step.tres_usage_total.get("cpu")
        if ceiling is not None and step.total_cpu is not None and step.total_cpu > ceiling:
            step.total_cpu = None
        step.gpu_util_ave = step.tres_usage_ave.get("gres/gpuutil")
        step.gpu_util_max = step.tres_usage_max.get("gres/gpuutil")
        step.gpu_util_total = step.tres_usage_total.get("gres/gpuutil")
        if any("gres/gpuutil" in usage for usage in usages):
            step.notes.append("gpuutil may be summed over GPUs and tasks; not percent busy")
        for attr, usage in (("gpu_mem_max", step.tres_usage_max),
                            ("gpu_mem_total", step.tres_usage_total)):
            if "gres/gpumem" in usage:
                setattr(step, attr, int(usage["gres/gpumem"]))
        steps.append(step)
    return steps


def fetch_steps(runner, job: Job) -> StepReport:
    if not job.job_id.isdigit():
        return StepReport(Status.EMPTY, detail="not a numeric job id")
    if job.state is not None and not (job.holds_resources or job.state == "SUSPENDED"):
        return StepReport(Status.EMPTY, detail="live step statistics exist only while running")
    result = runner.run(queries.steps(job.job_id))
    if not result.ok:
        return StepReport(result.status, detail=result.reason)
    steps = parse_steps(result.stdout, job)
    return StepReport(result.status if steps else Status.EMPTY, steps,
                      result.reason or ("" if steps else "no steps reported"))


def fetch_batch_script(runner, job_id: str) -> Fetched:
    """Status ok, not_stored (Slurm answered: no copy kept), denied (no record
    returned to this invoker), or the runner's failure status."""
    if not (job_id.isascii() and job_id.isdigit()):
        return Fetched(Status.EMPTY, detail="not a numeric job id")
    result = runner.batch_script(job_id)
    if not result.ok:
        return Fetched(result.status, detail=result.reason)
    body = parse.batch_script_body(result.stdout)
    if body is None:
        if result.status == Status.PARTIAL:
            return Fetched(Status.DENIED, detail=result.reason)
        return Fetched(Status.DENIED, detail=result.reason or parse.SCRIPT_NOT_VISIBLE)
    if not body.strip():
        return Fetched(Status.NOT_STORED, detail=parse.SCRIPT_NOT_STORED)
    return Fetched(result.status, clean_multiline(body), result.reason)


def fetch_submit_line(runner, job_id: str) -> Fetched:
    if not (job_id.isascii() and job_id.isdigit()):
        return Fetched(Status.EMPTY, detail="not a numeric job id")
    delimiter = "z" + secrets.token_hex(5) + "z"
    result = runner.run(queries.submit_line(job_id, delimiter))
    if not result.ok:
        return Fetched(result.status, detail=result.reason)
    records = []  # type: List[List[str]]
    for line in result.lines():
        raw_id, found, rest = line.partition(delimiter)
        if found:
            records.append([raw_id, rest])
        elif records:
            records[-1][1] += "\n" + line
    for raw_id, value in records:
        if raw_id == job_id and value.strip():
            return Fetched(result.status, clean_multiline(value), result.reason)
    return Fetched(Status.EMPTY, detail=result.reason or "not recorded, or not visible to you")
