"""The job table, built so that job-controlled text cannot forge anything.

1. Identity (id, user, uid, account, QOS, partition, state, ...) and every free
   text field come from squeue with random delimiters. This is authoritative.
2. ``scontrol -d show job`` only decorates those jobs (TRES, per-node GPUs,
   indices). Its free text can forge whole lines, so a record is used only if
   its header equals what squeue said (id, exact name, uid, submit time), it is
   the only such record, and the fields are read before the free-text tail.
   Otherwise ``detail_status`` is ``ambiguous`` or ``missing`` and the detail
   stays None.

The tail (submit line, command, comments) is never parsed or kept here.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

from .. import parse
from ..model import Allocation, ClusterInfo, Job
from ..sanitize import clean, clean_id
from .queries import QUEUE_FIELDS

_HEADER = re.compile(
    r"^JobId=([0-9]+) (?:(?:ArrayJobId|ArrayTaskId|ArrayTaskThrottle|HetJobId|HetJobOffset"
    r"|HetJobIdSet|PackJobId|PackJobOffset|PackJobIdSet)=[^ \n]+ )*JobName=", re.M)
_IDENTITY = re.compile(r"(?:SLUID=\S+ )?UserId=[^=\n]*\(([0-9]+)\) GroupId=")
_SUBMITTED = re.compile(r" SubmitTime=(\S+)")
_KEY_LIKE = re.compile(r"(?:^|\s)[A-Z][\w/:.*-]*=|\n")
_ALLOCATION = re.compile(r" Nodes=(\S+) CPU_IDs=(\S*) Mem=(\S*) (GRES|GRES_IDX)=(\S*)")

Row = Dict[str, str]


def parse_queue(lines: Iterable[str], frame: parse.Frame) -> Tuple[List[Row], int]:
    """(rows, how many records were unreadable)."""
    names = [name for name, _ in QUEUE_FIELDS]
    rows, bad = [], 0
    for values in frame.records(lines, len(names)):
        if values is None:
            bad += 1
            continue
        row = dict(zip(names, values))
        row["job_id"] = row["job_id"].strip()
        if row["job_id"].isascii() and row["job_id"].isdigit():
            rows.append(row)
        else:
            bad += 1
    return rows, bad


def _seconds(text: str) -> Optional[int]:
    value = parse.parse_duration(text)
    return None if value is None else int(value)


def _text(value: str) -> Optional[str]:
    return clean(parse.optional_text(value)) or None


def _ident(value: str) -> Optional[str]:
    return clean_id(parse.optional_text(value))


def build_jobs(rows: List[Row], detail_text: Optional[str], cluster: ClusterInfo,
               has_gpus: Optional[bool] = None) -> List[Job]:
    """``has_gpus`` False means the cluster has none, so every GPU count is zero."""
    groups = {}  # type: Dict[str, List[Row]]
    for row in rows:
        groups.setdefault(row["job_id"], []).append(row)
    details, ambiguous = _match_details(detail_text, groups)
    jobs = []
    for job_id, group in groups.items():
        job = _base_job(job_id, group)
        detail = details.get(job_id)
        if detail is not None:
            job.detail_status = "ok"
        elif job_id in ambiguous:
            job.detail_status = "ambiguous"
            job.notes.append("scheduler detail ignored: job text makes it ambiguous")
        elif detail_text is not None:
            job.detail_status = "missing"
        _resources(job, group[0], detail or "", cluster, has_gpus)
        jobs.append(job)
    return jobs


def _base_job(job_id: str, group: List[Row]) -> Job:
    row = group[0]
    job = Job(job_id=job_id, display_id=clean(row["display"]))
    task = parse.optional_text(row["array_task"])
    if task:
        job.array_job_id, job.array_task = clean(row["array_job"]), clean(task)
        tasks = set()
        for each in group:
            tasks.update(parse.parse_ranges(each["array_task"].split("%")[0]) or ())
        if len(tasks) > 1:
            job.task_count = len(tasks)
            job.array_task = parse.compress_ranges(tasks)
            job.display_id = "%s_[%s]" % (job.array_job_id, job.array_task)
    het = re.fullmatch(r"([0-9]+)\+([0-9]+)", row["display"].strip())
    if het:
        job.het_job_id, job.het_offset = het.group(1), int(het.group(2))

    job.raw_name = row["name"]
    job.name = clean(row["name"])
    job.uid = parse.parse_int(row["uid"])
    job.user = _ident(row["user"])
    job.account = _ident(row["account"])
    job.qos = _ident(row["qos"])
    job.partition = _ident(",".join(dict.fromkeys(
        part for each in group for part in each["partition"].split(","))))
    job.reservation = _ident(row["reservation"])
    job.wckey = _ident(row["wckey"])
    job.state = clean(parse.parse_state(row["state"])) or None
    job.reason = _text(row["reason"])
    job.priority = parse.parse_int(row["priority"])
    job.nice = parse.parse_int(row["nice"])
    job.dependency = _text(row["dependency"])
    job.comment = _text(row["comment"])
    job.submit = parse.parse_time(row["submit"])
    job.start = parse.parse_time(row["start"])
    job.end = parse.parse_time(row["end"])
    job.elapsed = _seconds(row["elapsed"])
    job.time_limit = _seconds(row["limit"])
    job.time_unlimited = parse.is_unlimited(row["limit"])
    job.node_count = parse.parse_count(row["node_count"])
    job.nodes = [clean_id(node) or "" for node in parse.expand_hostlist(row["nodes"])]
    job.batch_host = _ident(row["batch_host"])
    job.gres_request = _text(row["gres"])
    job.raw_workdir = parse.optional_text(row["workdir"])
    job.workdir = _text(row["workdir"])
    job.command = _text(row["command"])
    return job


def _match_details(text: Optional[str], groups: Dict[str, List[Row]]
                   ) -> Tuple[Dict[str, str], set]:
    """({job id: trusted middle of its record}, ambiguous ids)."""
    found, ambiguous = {}, set()  # type: Dict[str, str], set
    rejected = set()
    if not text:
        return found, ambiguous
    for header in _HEADER.finditer(text):
        job_id = header.group(1)
        group = groups.get(job_id)
        if not group or job_id in ambiguous:
            continue
        row = group[0]
        name = row["name"] + " "
        if not text.startswith(name, header.end()):
            continue
        body = header.end() + len(name)
        identity = _IDENTITY.match(text, body)
        if not identity or identity.group(1) != row["uid"].strip():
            continue
        line_end = text.find("\n", body)
        line_end = len(text) if line_end < 0 else line_end
        tail = text.find(" Features=", body, line_end)  # free text follows Features
        shaped = [row[key] for key in ("account", "qos", "wckey", "reason", "dependency",
                                       "reservation")]
        if tail < 0 or any(_KEY_LIKE.search(value) for value in shaped):
            rejected.add(job_id)
            continue
        middle = text[body:tail]
        submitted = _SUBMITTED.search(middle)
        if parse.parse_time(submitted.group(1) if submitted else "") != parse.parse_time(
                row["submit"]):
            continue
        if job_id in found:  # two plausible records: one of them is forged
            del found[job_id]
            ambiguous.add(job_id)
        else:
            found[job_id] = middle
    return found, ambiguous | (rejected - set(found))


def _keep_completing(job: Job) -> None:
    """A COMPLETING job's detail still lists every node but no longer its GPUs.

    Keep only nodes it has not left, and restore the GPU count where it is
    certain; elsewhere it is unknown rather than zero.
    """
    everywhere = job.allocations
    if not everywhere or any(each.gpus for each in everywhere):
        remaining = everywhere
    else:
        requested = [g for g in parse.parse_gres(job.gres_request) if g.name == "gpu"]
        per_node = parse.gpus_from_gres(requested)[0] if requested else None
        total = job.gpus_alloc
        for each in everywhere:
            if total == 0:
                each.gpus = 0
                continue
            each.gpu_indices = None  # already released
            if total is not None and len(everywhere) == 1:
                each.gpus, each.gpu_types = total, dict(job.gpu_types)
            elif per_node is not None and per_node * len(everywhere) == total:
                each.gpus = per_node
            else:
                each.gpus = None
        remaining = everywhere
    if job.nodes:
        job.allocations = [each for each in remaining if each.node in job.nodes]


def _allocations(middle: str, mem_tracked: Optional[bool]) -> List[Allocation]:
    start = middle.find(" Socks/Node=")
    if start < 0:
        return []
    stop = middle.find(" MinCPUsNode=", start)
    result = {}  # type: Dict[str, Allocation]
    region = middle[start:] if stop < 0 else middle[start:stop]
    for hosts, cpu_ids, mem_text, key, gres_text in _ALLOCATION.findall(region):
        entries = parse.parse_gres(gres_text)
        if key == "GRES_IDX":  # older Slurm lists only the indices, not a count
            for entry in entries:
                entry.count = len(entry.indices) if entry.indices is not None else None
        gpus, types = parse.gpus_from_gres(entries)
        cpus = parse.parse_ranges(cpu_ids)
        memory = parse.parse_memory(mem_text)
        if mem_tracked is False or (mem_tracked is None and not memory):
            memory = None
        indices = [] if gpus == 0 else parse.gpu_indices(entries)
        for host in parse.expand_hostlist(hosts):
            node = clean_id(host) or ""
            result[node] = Allocation(
                node, len(cpus) if cpus is not None else None, clean(cpu_ids) or None, memory,
                gpus, {clean(k): v for k, v in types.items()},
                list(indices) if indices is not None else None)
    return list(result.values())


def _resources(job: Job, row: Row, middle: str, cluster: ClusterInfo,
               has_gpus: Optional[bool]) -> None:
    kv = parse.key_values(middle)
    combined = kv.get("TRES")  # before ReqTRES/AllocTRES existed
    req = parse.parse_tres(kv.get("ReqTRES", combined))
    alloc = parse.parse_tres(kv.get("AllocTRES", None if job.pending else combined))
    job.tres_req = {clean(k): v for k, v in req.items()}
    job.tres_alloc = {clean(k): v for k, v in alloc.items()}
    job.restarts = parse.parse_int(kv.get("Restarts"))
    job.eligible = parse.parse_time(kv.get("EligibleTime"))
    if not job.active:
        job.exit_code, job.exit_signal = parse.parse_exit_code(kv.get("ExitCode"))
    if job.holds_resources:
        job.allocations = _allocations(middle, cluster.mem_tracked)
    if job.node_count is None:
        job.node_count = parse.parse_count(kv.get("NumNodes", "").split("-")[0])

    queue_cpus = parse.parse_count(row["cpus"].split("-")[0])
    if "cpu" in req:
        job.cpus_req = int(req["cpu"])
    elif job.pending:
        job.cpus_req = queue_cpus
    if "mem" in req:
        job.mem_req = int(req["mem"])
    job.gpus_req, req_types = parse.gpus_from_tres(req)
    if job.gpus_req is None:
        if has_gpus is False or (req and cluster.gpu_accounted):
            job.gpus_req = 0
        elif job.node_count:  # untracked GPUs: only a per-node request is countable
            entries = [g for g in parse.parse_gres(row["gres"]) if g.name == "gpu"]
            per_node = parse.gpus_from_gres(entries)[0] if entries else None
            if per_node is not None:
                job.gpus_req = per_node * job.node_count

    if job.pending:
        job.gpu_types = {clean(k): v for k, v in req_types.items()}
        return
    if "cpu" in alloc:
        job.cpus_alloc = int(alloc["cpu"])
    else:
        job.cpus_alloc = queue_cpus
    if "mem" in alloc and cluster.mem_tracked is not False:
        job.mem_alloc = int(alloc["mem"])
    job.gpus_alloc, types = parse.gpus_from_tres(alloc)
    if job.gpus_alloc is None:
        counts = [each.gpus for each in job.allocations]
        if has_gpus is False or (alloc and cluster.gpu_accounted):
            job.gpus_alloc = 0
        elif counts and None not in counts and (job.state != "COMPLETING" or any(counts)):
            job.gpus_alloc = sum(count for count in counts if count is not None)
    if not types:
        for each in job.allocations:
            for kind, count in each.gpu_types.items():
                types[kind] = types.get(kind, 0) + count
    job.gpu_types = {clean(k): v for k, v in types.items()}
    if job.state == "COMPLETING":
        _keep_completing(job)
