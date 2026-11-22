"""Jobs, their attempts (one per start/requeue) and steps."""

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional

# States in which a job holds CPUs and GPUs. COMPLETING still holds them.
HOLDING_STATES = frozenset(
    ("RUNNING", "COMPLETING", "CONFIGURING", "STOPPED", "SIGNALING", "RESIZING"))
PENDING_STATES = frozenset(
    ("PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD", "SPECIAL_EXIT"))
# Live but neither holding nor waiting. SUSPENDED gives its CPUs back.
OTHER_LIVE_STATES = frozenset(("SUSPENDED", "STAGE_OUT"))


class RoleKey(NamedTuple):
    account: Optional[str]
    qos: Optional[str]
    partition: Optional[str]


@dataclass
class Allocation:
    """What a job holds on one node."""

    node: str
    cpus: Optional[int] = None
    cpu_ids: Optional[str] = None
    mem: Optional[int] = None  # bytes
    gpus: Optional[int] = None
    gpu_types: Dict[str, int] = field(default_factory=dict)
    gpu_indices: Optional[List[int]] = None


@dataclass
class Step:
    job_id: str
    step_id: str  # "batch", "extern", "interactive", "0", ...
    name: Optional[str] = None
    state: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    elapsed: Optional[int] = None  # seconds
    exit_code: Optional[int] = None
    exit_signal: Optional[int] = None
    ntasks: Optional[int] = None
    nodes: List[str] = field(default_factory=list)
    ave_cpu: Optional[float] = None  # seconds per task
    total_cpu: Optional[float] = None  # seconds
    ave_rss: Optional[int] = None  # bytes per task
    max_rss: Optional[int] = None  # bytes, largest task
    # gres/gpuutil as Slurm reports it. Depending on site and version a task's value
    # may already be summed over its GPUs, so this can exceed 100 and is not "percent busy".
    gpu_util_ave: Optional[float] = None  # average over tasks
    gpu_util_max: Optional[float] = None  # largest task
    gpu_util_total: Optional[float] = None  # summed over tasks
    gpu_mem_max: Optional[int] = None  # bytes, largest task
    gpu_mem_total: Optional[int] = None  # bytes, summed over tasks
    tres_alloc: Dict[str, float] = field(default_factory=dict)
    tres_usage_ave: Dict[str, float] = field(default_factory=dict)
    tres_usage_max: Dict[str, float] = field(default_factory=dict)
    tres_usage_total: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class Attempt:
    """One run of a job. A requeue starts a new attempt under the same job id."""

    job_id: str
    number: Optional[int] = None  # 0-based, when known
    state: Optional[str] = None
    submit: Optional[int] = None
    eligible: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    elapsed: Optional[int] = None
    exit_code: Optional[int] = None
    exit_signal: Optional[int] = None
    reason: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    partition: Optional[str] = None
    nodes: List[str] = field(default_factory=list)
    cpus: Optional[int] = None
    gpus: Optional[int] = None
    mem: Optional[int] = None
    tres_alloc: Dict[str, float] = field(default_factory=dict)
    steps: List[Step] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class Job:
    """One Slurm job record. An unsplit pending array is one record of task_count tasks."""

    job_id: str  # numeric id
    display_id: str = ""  # 12, 12_3, 12_[4-9], 12+1
    array_job_id: Optional[str] = None
    array_task: Optional[str] = None
    task_count: int = 1
    het_job_id: Optional[str] = None
    het_offset: Optional[int] = None

    name: Optional[str] = None
    user: Optional[str] = None
    uid: Optional[int] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    partition: Optional[str] = None  # comma list while pending on several
    reservation: Optional[str] = None
    wckey: Optional[str] = None
    state: Optional[str] = None
    reason: Optional[str] = None
    priority: Optional[int] = None
    nice: Optional[int] = None
    dependency: Optional[str] = None
    comment: Optional[str] = None

    submit: Optional[int] = None  # epoch seconds
    eligible: Optional[int] = None
    start: Optional[int] = None  # an estimate while pending
    end: Optional[int] = None  # an estimate until finished
    elapsed: Optional[int] = None
    time_limit: Optional[int] = None  # seconds
    time_unlimited: bool = False
    restarts: Optional[int] = None
    exit_code: Optional[int] = None
    exit_signal: Optional[int] = None

    node_count: Optional[int] = None
    nodes: List[str] = field(default_factory=list)
    batch_host: Optional[str] = None
    cpus_alloc: Optional[int] = None
    gpus_alloc: Optional[int] = None
    mem_alloc: Optional[int] = None  # bytes; None when untracked
    cpus_req: Optional[int] = None  # per array task
    gpus_req: Optional[int] = None
    mem_req: Optional[int] = None
    gpu_types: Dict[str, int] = field(default_factory=dict)
    gres_request: Optional[str] = None
    tres_req: Dict[str, float] = field(default_factory=dict)
    tres_alloc: Dict[str, float] = field(default_factory=dict)
    allocations: List[Allocation] = field(default_factory=list)

    workdir: Optional[str] = None  # cleaned, for display
    command: Optional[str] = None  # cleaned, for display
    raw_workdir: Optional[str] = field(default=None, repr=False)  # never display
    raw_name: Optional[str] = field(default=None, repr=False)  # never display

    detail_status: str = "unknown"  # ok, missing, ambiguous (see live.jobs)
    attempts: List[Attempt] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def holds_resources(self) -> bool:
        return self.state in HOLDING_STATES

    @property
    def pending(self) -> bool:
        return self.state in PENDING_STATES

    @property
    def active(self) -> bool:
        return self.holds_resources or self.pending or self.state in OTHER_LIVE_STATES

    @property
    def role(self) -> RoleKey:
        return RoleKey(self.account, self.qos, self.partition)
