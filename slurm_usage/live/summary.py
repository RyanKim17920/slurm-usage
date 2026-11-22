"""Totals, group-by and health, derived purely from nodes and jobs."""

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from ..model import (Availability, CapacityCheck, GpuBlock, Health, Job, LoadOutlier, Node,
                     Resource, Totals, Usage)
from .nodes import is_future


def add(values: Iterable[Optional[int]]) -> Optional[int]:
    """Sum, or None if any addend is unknown."""
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def _split(node: Node, total: Optional[int], alloc: Optional[int], free: Optional[int],
           shared: Optional[int] = 0, completing: Optional[int] = 0) -> Resource:
    part = Resource(total=total, alloc=alloc, shared=shared, completing=completing)
    if total == 0:
        return Resource(0, 0, 0, 0, 0, 0, 0)
    if None in (total, alloc, shared, completing) or node.availability == Availability.UNKNOWN:
        return part
    idle = max(0, total - alloc - shared - completing)  # type: ignore[operator]
    if node.availability == Availability.RESERVED:
        part.free, part.unavailable, part.reserved = 0, 0, idle
    elif node.availability == Availability.UNAVAILABLE:
        part.free, part.unavailable, part.reserved = 0, idle, 0
    elif free is not None:
        usable = min(free, idle)
        # schedulable but unusable remainder: specialized cores or memory
        part.free, part.unavailable, part.reserved = usable, idle - usable, 0
    return part


def _combine(parts: Sequence[Resource]) -> Resource:
    return Resource(**{name: add(getattr(part, name) for part in parts)
                       for name in ("total", "alloc", "free", "unavailable", "reserved",
                                    "shared", "completing")})


def totals(nodes: Optional[List[Node]]) -> Totals:
    physical = [node for node in nodes or () if not is_future(node)]
    if not physical:
        return Totals()  # nothing visible is unknown, not a cluster of size zero
    result = Totals(nodes=len(physical), nodes_schedulable=sum(n.schedulable for n in physical))
    result.cpus = _combine([_split(n, n.cpus_total, n.cpus_alloc, n.cpus_free, 0,
                                   n.cpus_completing) for n in physical])
    result.mem = _combine([_split(n, n.mem_total, n.mem_alloc, n.mem_free, 0,
                                  n.mem_completing) for n in physical])
    result.gpus = _combine(
        [_split(n, n.gpus_total, n.gpus_alloc, n.gpus_free, n.gpus_shared, n.gpus_completing)
         for n in physical])

    by_type = {}  # type: Dict[str, List[Resource]]
    for node in physical:
        free = node.gpus_free_by_type()
        only = len(node.gpu_types) == 1
        for kind, count in node.gpu_types.items():
            if only:
                part = _split(node, count, node.gpus_alloc, node.gpus_free, node.gpus_shared,
                              node.gpus_completing)
            elif node.gpu_types_alloc is None or node.gpus_shared or node.gpus_completing:
                part = Resource(total=count)
            else:
                part = _split(node, count, node.gpu_types_alloc.get(kind, 0),
                              free.get(kind, 0) if free is not None else None)
            by_type.setdefault(kind, []).append(part)
    result.gpu_types = {kind: _combine(parts) for kind, parts in sorted(by_type.items())}
    return result


def mark_completing(nodes: Optional[List[Node]], jobs: Optional[List[Job]]) -> None:
    """Set Node.*_completing.

    Slurm versions differ on whether a node's allocation still includes a
    COMPLETING job, so only what the node does not already count is added:
    min(completing jobs' share, all jobs on the node - the node's allocation).
    """
    if not nodes or not jobs:
        return
    held = {}  # type: Dict[str, list]
    affected = set()
    for job in jobs:
        if not job.holds_resources:
            continue
        for each in job.allocations:
            held.setdefault(each.node, []).append((job.state == "COMPLETING", each))
            if job.state == "COMPLETING":
                affected.add(each.node)
    for node in nodes:
        if node.name not in affected:
            continue
        for what in ("cpus", "mem", "gpus"):
            alloc = getattr(node, what + "_alloc")
            if alloc is None:
                continue  # free is unknown anyway
            pairs = [(done, getattr(each, what)) for done, each in held[node.name]]
            finishing = add(value for done, value in pairs if done)
            if finishing:
                everything = add(value for _, value in pairs)
                finishing = (None if everything is None
                             else min(finishing, max(0, everything - alloc)))
            setattr(node, what + "_completing", finishing)


def reason_code(reason: Optional[str]) -> str:
    """"ReqNodeNotAvail, UnavailableNodes:..." groups as "ReqNodeNotAvail"."""
    return re.split(r"[,:]", reason or "None", maxsplit=1)[0].strip()[:80]


_KEYS = {
    "user": lambda job: job.user,
    "account": lambda job: job.account,
    "qos": lambda job: job.qos,
    "partition": lambda job: job.partition,
    "role": lambda job: job.role,
    "state": lambda job: job.state,
    "name": lambda job: job.name,
    "reason": lambda job: reason_code(job.reason),
    "total": lambda job: "total",
}  # type: Dict[str, Callable[[Job], object]]


def group_by(jobs: Iterable[Job], key: Union[str, Callable[[Job], object]]) -> List[Usage]:
    """Usage per user, account, qos, partition, role, state, name, reason, total,
    node, or any callable. Finished jobs are ignored; "node" counts only what
    each job holds on that node. Largest GPU user first.
    """
    buckets = {}  # type: Dict[object, list]
    get = key if callable(key) else (None if key == "node" else _KEYS[key])
    for job in jobs:
        if not job.active:
            continue
        if get is not None:
            buckets.setdefault(get(job), []).append(
                (job, job.cpus_alloc, job.gpus_alloc, job.mem_alloc))
        elif job.holds_resources:
            if job.allocations:
                for each in job.allocations:
                    buckets.setdefault(each.node, []).append(
                        (job, each.cpus, each.gpus, each.mem))
            else:
                single = len(job.nodes) == 1
                for node in job.nodes:
                    buckets.setdefault(node, []).append(
                        (job, job.cpus_alloc, job.gpus_alloc, job.mem_alloc) if single
                        else (job, None, None, None))
    result = []
    for bucket, entries in buckets.items():
        held = [entry for entry in entries if entry[0].holds_resources]
        waiting = [entry[0] for entry in entries if entry[0].pending]
        result.append(Usage(
            key=bucket, running=len(held), pending=sum(job.task_count for job in waiting),
            other=len(entries) - len(held) - len(waiting),
            cpus=add(entry[1] for entry in held), gpus=add(entry[2] for entry in held),
            mem=add(entry[3] for entry in held),
            pending_cpus=add(_times(job.cpus_req, job.task_count) for job in waiting),
            pending_gpus=add(_times(job.gpus_req, job.task_count) for job in waiting),
            pending_mem=add(_times(job.mem_req, job.task_count) for job in waiting)))
    result.sort(key=lambda u: (-(u.gpus or 0), -(u.cpus or 0), -u.running, -u.pending,
                               str(u.key)))
    return result


def _times(value: Optional[int], count: int) -> Optional[int]:
    return None if value is None else value * count


def load_outliers(nodes: Iterable[Node]) -> List[LoadOutlier]:
    """Hints only: load average counts runnable and IO-blocked threads, not busy CPUs."""
    found = []
    for node in nodes:
        load, alloc, size = node.cpu_load, node.cpus_alloc, node.cpus_total
        if load is None or alloc is None or not size:
            continue
        if node.state == "DOWN" or "NOT_RESPONDING" in node.state_flags:
            continue  # the load of a node that is not reporting is stale
        if load > max(alloc * 1.25, alloc + max(2, 0.1 * size)):
            found.append(LoadOutlier(node.name, "load_above_alloc", load, alloc))
        elif alloc >= max(4, 0.25 * size) and load < 0.1 * alloc:
            found.append(LoadOutlier(node.name, "load_below_alloc", load, alloc))
    return found


def _larger(node: Node, count: int, best: Optional[GpuBlock]) -> Optional[GpuBlock]:
    if count <= 0:
        return best
    if best and (count, node.cpus_free or 0) <= (best.gpus, best.cpus_free or 0):
        return best
    return GpuBlock(node.name, count, node.cpus_free, node.mem_free)


def health(nodes: Optional[List[Node]], jobs: Optional[List[Job]]) -> Health:
    result = Health()
    if jobs is not None:
        waiting = group_by([job for job in jobs if job.pending], "reason")
        result.pending_by_reason = sorted(waiting, key=lambda u: (-u.pending, str(u.key)))
    if not nodes:
        return result
    for node in nodes:
        # the base state with its flags (IDLE+DRAIN, MIXED+NOT_RESPONDING): Slurm keeps
        # DRAIN, DOWN-ish and maintenance flags apart from the base word
        label = "+".join([node.state or "unknown"] + list(node.state_flags or ()))
        result.node_states[label] = result.node_states.get(label, 0) + 1
    present = [node for node in nodes if not is_future(node)]
    result.problem_nodes = [n for n in present if n.availability == Availability.UNAVAILABLE]
    result.load_outliers = load_outliers(present)

    with_gpus = [node for node in present if node.gpus_total != 0]
    result.free_gpus_by_node = [(node.name, node.gpus_free) for node in with_gpus]
    result.gpus_stranded_cpu = add(node.gpus_stranded_cpu for node in with_gpus)
    # meaningless unless Slurm tracks memory, so unknown then
    result.gpus_stranded_mem = add(node.gpus_stranded_mem for node in with_gpus)
    for node in with_gpus:
        startable = node.gpus_startable
        if not startable:
            continue
        result.largest_gpu_block = _larger(node, startable, result.largest_gpu_block)
        for kind, count in (node.gpus_free_by_type() or {}).items():
            best = _larger(node, min(count, startable),
                           result.largest_gpu_block_by_type.get(kind))
            if best:
                result.largest_gpu_block_by_type[kind] = best
    return result


def capacity_checks(live: Totals, accounting: Optional[Dict[str, float]]) -> List[CapacityCheck]:
    """Live capacity is the deduplicated nodes'; slurmdbd's is only compared."""
    if accounting is None:
        return []
    rows = []
    for metric, value in (("gres/gpu", live.gpus.total), ("cpu", live.cpus.total),
                          ("mem", live.mem.total), ("node", live.nodes)):
        recorded = accounting.get(metric)
        if recorded is not None:
            rows.append(CapacityCheck(metric, value, int(recorded)))
    return rows
