"""Check the two independent views of allocated GPUs: nodes and jobs."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..model import Snapshot
from .nodes import is_future
from .summary import add


@dataclass
class NodeDifference:
    node: str
    node_says: Optional[int]
    jobs_say: int
    causes: List[str] = field(default_factory=list)


@dataclass
class Reconciliation:
    node_total: Optional[int] = None  # capacity
    node_alloc: Optional[int] = None
    node_completing: Optional[int] = None  # held, but no longer in the nodes' allocation
    jobs_alloc: Optional[int] = None
    per_node_alloc: Optional[int] = None  # sum of job per-node detail
    differences: List[NodeDifference] = field(default_factory=list)
    causes: List[str] = field(default_factory=list)  # empty with a difference = unexplained

    @property
    def difference(self) -> Optional[int]:
        if None in (self.node_alloc, self.node_completing, self.jobs_alloc):
            return None
        return self.node_alloc + self.node_completing - self.jobs_alloc

    @property
    def consistent(self) -> bool:
        return self.difference == 0 and not self.differences

    @property
    def explained(self) -> bool:
        return self.consistent or (
            bool(self.causes) and all(diff.causes for diff in self.differences))


def reconcile_gpus(snapshot: Snapshot) -> Reconciliation:
    result = Reconciliation(node_total=snapshot.totals.gpus.total,
                            node_alloc=snapshot.totals.gpus.alloc,
                            node_completing=snapshot.totals.gpus.completing)
    if snapshot.nodes is None or snapshot.jobs is None:
        result.causes.append("nodes or jobs are not available")
        return result
    held = [job for job in snapshot.jobs if job.holds_resources]
    if all(job.gpus_alloc is not None for job in held):
        result.jobs_alloc = sum(job.gpus_alloc or 0 for job in held)
    else:
        result.causes.append("some jobs' GPU counts are unknown")

    on_node = {}  # type: Dict[str, int]
    notes = {}  # type: Dict[str, List[str]]
    for job in snapshot.jobs:
        if job.holds_resources:
            for each in job.allocations:
                on_node[each.node] = on_node.get(each.node, 0) + (each.gpus or 0)
            if job.state == "COMPLETING":
                cause = "completing jobs release nodes one by one"
            elif not job.allocations or any(a.gpus is None for a in job.allocations):
                cause = "jobs without per-node detail"
            else:
                continue
        elif job.state == "SUSPENDED":
            cause = "suspended jobs may keep GPUs"
        else:
            continue
        for node in job.nodes or [a.node for a in job.allocations]:
            notes.setdefault(node, []).append(cause)
    result.per_node_alloc = sum(on_node.values())
    for job in held:
        if job.gpus_alloc and job.allocations:
            parts = [each.gpus for each in job.allocations]
            if None not in parts and sum(parts) != job.gpus_alloc:
                result.causes.append(
                    "%s jobs' TRES totals differ from their per-node detail" % job.state)

    limited = any(w.code == "visibility_limited" for w in snapshot.warnings)
    for node in snapshot.nodes:
        if is_future(node):
            continue
        jobs_say = on_node.pop(node.name, 0)
        expected = add([node.gpus_alloc, node.gpus_completing])
        if expected == jobs_say:
            continue
        causes = sorted(set(notes.get(node.name, ())))
        if expected is None:
            causes.append("allocation on the node is unknown")
        elif limited and expected > jobs_say:
            causes.append("other users' jobs are hidden (PrivateData)")
        if node.gpus_shared:
            causes.append("shared GPUs (shard/MPS)")
        result.differences.append(NodeDifference(node.name, node.gpus_alloc, jobs_say, causes))
        result.causes.extend(causes)
    for name, count in on_node.items():
        if count:
            result.differences.append(
                NodeDifference(name, None, count, ["node is not visible to the invoker"]))
            result.causes.append("node is not visible to the invoker")
    result.causes = list(dict.fromkeys(result.causes))
    return result
