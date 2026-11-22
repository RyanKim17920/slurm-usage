"""sacct text to normalized records. No I/O.

Any user can put newlines, separators or a whole forged record in a job name or
a path. So every command gets its own unpredictable delimiter, free text goes
last, and records are framed by counting delimiters rather than lines.
"""

import secrets
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

from ..model import HOLDING_STATES, PENDING_STATES
from ..parse import (gpus_from_tres, parse_exit_code, is_unlimited, memory_basis, optional_text, parse_duration,
                     parse_int, parse_memory, parse_state, parse_time, parse_tres)
from ..sanitize import clean, clean_id

NONE = "(none)"  # group key for a value that is absent
UNLIMITED = -1
NAME_LIMIT = 200

# What an open allocation must be in for "End=Unknown" to mean "still going".
ACTIVE_STATES = frozenset(HOLDING_STATES | {"SUSPENDED", "STAGE_OUT"})

OUTCOMES = ("completed", "preempted", "failed", "timeout", "cancelled", "node_fail", "running",
            "requeued", "other")
_OUTCOME_OF = {
    "COMPLETED": "completed",
    "PREEMPTED": "preempted",
    "FAILED": "failed",
    "OUT_OF_MEMORY": "failed",
    "LAUNCH_FAILED": "failed",
    "TIMEOUT": "timeout",
    "DEADLINE": "timeout",
    "CANCELLED": "cancelled",
    "NODE_FAIL": "node_fail",
    "BOOT_FAIL": "node_fail",
    "RECONFIG_FAIL": "node_fail",
    "REQUEUED": "requeued",
    "REQUEUE_HOLD": "requeued",
    "REQUEUE_FED": "requeued",
    "SPECIAL_EXIT": "requeued",
}

ALLOCATION_FIELDS = (
    "DBIndex", "JobIDRaw", "Cluster", "JobID", "Submit", "Eligible", "Start", "End", "Suspended",
    "ElapsedRaw", "State", "AllocTRES", "ReqTRES", "ReqMem", "NNodes", "AllocCPUS",
    "TimelimitRaw", "Restarts", "ExitCode", "UID", "User", "Account", "QOS", "Partition", "NodeList",
    "JobName",  # user-controlled, so last
)
REQUIRED_FIELDS = ("JobIDRaw", "Start", "End", "State")
# Must never be requested in bulk or written to the cache.
NEVER_BULK = frozenset(("SubmitLine", "Comment", "AdminComment", "WorkDir", "StdIn"))


def outcome_of(state: str, ended: bool = True) -> str:
    if not ended:
        return "running" if state in ACTIVE_STATES else "other"
    return _OUTCOME_OF.get(state, "other")


def new_delimiter() -> str:
    return "\x1f" + secrets.token_hex(4) + "\x1f"


def split_records(lines: Iterable[str], separator: str,
                  width: int) -> Iterator[Optional[List[str]]]:
    """Yield field lists, or None for something that is not a whole record.

    The first field is always an id without newlines, so a line without a
    delimiter can only continue the record before it.
    """
    need = width - 1
    text, found = None, 0  # type: Optional[str], int
    for line in lines:
        if line.endswith("\n"):
            line = line[:-1]
        count = line.count(separator)
        if text is None:
            if count:
                text, found = line, count
            elif line.strip():
                yield None
        elif found < need or not count:
            text += "\n" + line
            found += count
        else:
            yield text.split(separator) if found == need else None
            text, found = line, count
    if text is not None:
        yield text.split(separator) if found == need else None


class Roles(NamedTuple):
    user: str
    uid: Optional[int]
    account: str
    qos: str
    partition: str
    state: str


class Record(NamedTuple):
    """One allocation attempt. ``start`` is None if nothing was ever allocated."""

    key: str  # accounting database index, else "job:submit"
    job: int
    label: str  # 12, 12_3, 12_[4-9], 12+1
    name: str
    roles: Roles
    nodes: str  # hostlist
    submit: Optional[int]
    eligible: Optional[int]
    start: Optional[int]
    end: Optional[int]  # None: accounting has not seen it end
    suspended: int
    cpus: Optional[int]
    gpus: Optional[int]
    mem: Optional[int]  # bytes
    limit: Optional[int]  # seconds, or UNLIMITED
    restarts: Optional[int]
    exit: str = ""  # "status:signal" as sacct prints it, or ""


@lru_cache(maxsize=8192)
def alloc_amounts(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """(cpus, gpus, memory) from AllocTRES; all None if the string is empty."""
    tres = parse_tres(text)
    if not tres:
        return None, None, None
    total, _ = gpus_from_tres(tres)
    cpus, mem = tres.get("cpu"), tres.get("mem")
    return (int(cpus) if cpus is not None else None,
            total or 0,  # no GPU key on a real allocation means none were allocated
            int(mem) if mem else None)


@lru_cache(maxsize=8192)
def requested_memory(req_tres: str, req_mem: str, cpus: Optional[int],
                     nodes: Optional[int]) -> Optional[int]:
    """Bytes. ReqTRES wins; old ReqMem is per cpu ("c") or per node ("n"). 0 means unknown."""
    total = parse_tres(req_tres).get("mem")
    if total:
        return int(total)
    size = parse_memory(req_mem)
    if not size:
        return None
    basis = memory_basis(req_mem)
    if basis:
        count = cpus if basis == "cpu" else nodes
        return size * count if count else None
    return size


@lru_cache(maxsize=4096)
def parse_roles(user: str, uid_text: str, account: str, qos: str, partition: str,
                state: str) -> Roles:
    uid = parse_int(uid_text)
    if user:
        name = clean_id(user)
    else:
        name = "uid:%d" % uid if uid is not None else NONE
    return Roles(name or NONE, uid, clean_id(account) or NONE, clean_id(qos) or NONE,
                 clean_id(partition) or NONE, clean(parse_state(state) or "UNKNOWN"))


@lru_cache(maxsize=1024)
def parse_limit(text: str) -> Optional[int]:
    if is_unlimited(text):
        return UNLIMITED
    minutes = parse_int(text)
    return minutes * 60 if minutes and minutes > 0 else None


@lru_cache(maxsize=4096)
def parse_nodes(text: str) -> str:
    value = optional_text(text)
    if not value or value.lower() == "none assigned":
        return ""
    return clean(value)


@lru_cache(maxsize=1024)
def parse_exit(text: str) -> str:
    status, signal = parse_exit_code(text)
    return "" if status is None else "%d:%d" % (status, signal or 0)


def record_key(index: Optional[str], job: object, submit: object) -> str:
    if index and index.isascii() and index.isdigit() and int(index):
        return index
    return "%s:%s" % (job, submit or 0)


def normalize(row: Dict[str, str], seen_at: Optional[int] = None) -> Optional[Record]:
    """A Record, or None for a step or an unreadable row. ``seen_at``: when it was fetched."""
    raw_job = row.get("JobIDRaw", "")
    if not (raw_job.isascii() and raw_job.isdigit()) or int(raw_job) >= 2 ** 63:
        return None
    if not parse_state(row.get("State")):
        return None
    job = int(raw_job)
    roles = parse_roles(row.get("User", ""), row.get("UID", ""), row.get("Account", ""),
                        row.get("QOS", ""), row.get("Partition", ""), row.get("State", ""))
    submit = parse_time(row.get("Submit"))
    start, end = parse_time(row.get("Start")), parse_time(row.get("End"))
    elapsed = parse_int(row.get("ElapsedRaw"))

    cpus, gpus, mem = alloc_amounts(row.get("AllocTRES", ""))
    if not cpus:
        cpus = parse_int(row.get("AllocCPUS")) or None

    if start is not None:
        if end is None and roles.state in PENDING_STATES and not elapsed:
            start = None  # a scheduler estimate
        elif ("AllocTRES" in row and not row["AllocTRES"] and not cpus and not elapsed
              and end is not None and end <= start):
            start = None  # cancelled while pending

    suspended = 0
    if start is not None:
        if end is not None and end < start:
            end = start
        reported = row.get("Suspended")
        if reported is not None:
            suspended = int(parse_duration(reported) or 0)
        elif end is not None and elapsed is not None:
            suspended = max(0, end - start - elapsed)
        if end is None and roles.state == "SUSPENDED" and seen_at and elapsed is not None:
            # Elapsed stops while suspended; the suspension still running is not in Suspended
            suspended = max(suspended, seen_at - start - elapsed)
        if not mem:
            mem = requested_memory(row.get("ReqTRES", ""), row.get("ReqMem", ""), cpus,
                                   parse_int(row.get("NNodes")))
    else:
        cpus = gpus = mem = None

    return Record(record_key(row.get("DBIndex"), job, submit), job,
                  clean(row.get("JobID") or raw_job, 80), clean(row.get("JobName"), NAME_LIMIT),
                  roles, parse_nodes(row.get("NodeList", "")) if start is not None else "",
                  submit, parse_time(row.get("Eligible")), start, end, suspended, cpus, gpus,
                  mem, parse_limit(row.get("TimelimitRaw", "")),
                  parse_int(row.get("Restarts")), parse_exit(row.get("ExitCode", "")))


def read_rows(lines: Iterable[str], separator: str,
              fields: Sequence[str]) -> Iterator[Optional[Dict[str, str]]]:
    for values in split_records(lines, separator, len(fields)):
        yield None if values is None else dict(zip(fields, values))
