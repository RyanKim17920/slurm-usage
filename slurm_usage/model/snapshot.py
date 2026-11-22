"""One consistent-enough picture of the cluster."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .job import Job
from .node import Node, Partition
from .roles import Limit, Qos, Role
from .status import Capability, Invoker
from .summary import CapacityCheck, Health, PriorityConfig, Totals, Warning_


@dataclass
class ClusterInfo:
    name: Optional[str] = None
    version: Optional[str] = None
    select_parameters: Optional[str] = None
    mem_tracked: Optional[bool] = None
    gpu_accounted: Optional[bool] = None  # gres/gpu is a tracked TRES
    has_accounting: Optional[bool] = None
    min_job_age: Optional[int] = None


@dataclass
class Snapshot:
    """Output of live.LiveCollector.snapshot().

    None for nodes, jobs, partitions, roles, qos or limits means the query failed;
    an empty list means it succeeded and found nothing. ``capabilities`` says why.
    """

    taken_at: float
    duration: float = 0.0
    cluster: ClusterInfo = field(default_factory=ClusterInfo)
    invoker: Invoker = field(default_factory=Invoker)
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    nodes: Optional[List[Node]] = None
    partitions: Optional[List[Partition]] = None
    jobs: Optional[List[Job]] = None  # includes recently finished; see Job.active
    roles: Optional[List[Role]] = None
    qos: Optional[List[Qos]] = None
    limits: Optional[List[Limit]] = None  # every limit, all scopes
    totals: Totals = field(default_factory=Totals)
    health: Health = field(default_factory=Health)
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    capacity: List[CapacityCheck] = field(default_factory=list)
    warnings: List[Warning_] = field(default_factory=list)

    @property
    def my_jobs(self) -> Optional[List[Job]]:
        if self.jobs is None or self.invoker.uid is None:
            return None
        return [job for job in self.jobs if job.uid == self.invoker.uid]

    @property
    def my_roles(self) -> Optional[List[Role]]:
        if self.roles is None or self.invoker.user is None:
            return None
        return [role for role in self.roles if role.user == self.invoker.user]

    def job(self, job_id: str) -> Optional[Job]:
        """Find by numeric id or display id."""
        for job in self.jobs or ():
            if job_id in (job.job_id, job.display_id):
                return job
        return None

    def node(self, name: str) -> Optional[Node]:
        for node in self.nodes or ():
            if node.name == name:
                return node
        return None
