"""Physical nodes, partitions and reservations."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class Availability:
    """Which bucket a node's *unallocated* capacity belongs to."""

    SCHEDULABLE = "schedulable"
    UNAVAILABLE = "unavailable"  # down, drain, fail, not responding, maint, no UP partition
    RESERVED = "reserved"
    UNKNOWN = "unknown"


@dataclass
class Node:
    """One physical node, however many partitions list it."""

    name: str
    state: Optional[str] = None  # base state, e.g. MIXED
    state_flags: List[str] = field(default_factory=list)
    availability: str = Availability.UNKNOWN
    unavailable_because: List[str] = field(default_factory=list)
    partitions: List[str] = field(default_factory=list)

    cpus_total: Optional[int] = None
    cpus_effective: Optional[int] = None  # total less specialized CPUs
    cpus_alloc: Optional[int] = None
    threads_per_core: Optional[int] = None
    cpu_load: Optional[float] = None  # load average, NOT utilization

    mem_total: Optional[int] = None  # bytes
    mem_schedulable: Optional[int] = None  # total less MemSpecLimit
    mem_alloc: Optional[int] = None  # bytes; None when Slurm does not track memory
    mem_os_free: Optional[int] = None  # FreeMem: excludes page cache, NOT available memory

    gpus_total: Optional[int] = None
    gpus_alloc: Optional[int] = None
    gpu_types: Dict[str, int] = field(default_factory=dict)
    gpu_types_alloc: Optional[Dict[str, int]] = None  # None = use by type is unknown
    gpu_indices_alloc: Optional[List[int]] = None
    gpus_shared: Optional[int] = None  # held through shard/MPS, so not free as whole GPUs
    default_cpus_per_gpu: Optional[int] = None  # only when all its partitions agree

    # Still held by visible COMPLETING jobs although Slurm has already taken them out
    # of the node's allocation. Not free until the job has finished completing.
    cpus_completing: Optional[int] = 0
    mem_completing: Optional[int] = 0
    gpus_completing: Optional[int] = 0

    features: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    reason_user: Optional[str] = None
    reason_time: Optional[int] = None
    boot_time: Optional[int] = None
    last_busy: Optional[int] = None
    tres: Dict[str, float] = field(default_factory=dict)
    tres_alloc: Dict[str, float] = field(default_factory=dict)

    @property
    def schedulable(self) -> bool:
        return self.availability == Availability.SCHEDULABLE

    @property
    def cpus_free(self) -> Optional[int]:
        """CPUs a new job could get now; 0 on an unavailable node."""
        if self.availability == Availability.UNKNOWN:
            return None
        if not self.schedulable:
            return 0
        usable = self.cpus_effective if self.cpus_effective is not None else self.cpus_total
        if usable is None or self.cpus_alloc is None or self.cpus_completing is None:
            return None
        return max(0, usable - self.cpus_alloc - self.cpus_completing)

    @property
    def mem_free(self) -> Optional[int]:
        """Unallocated memory as Slurm counts it; None when untracked."""
        if self.availability == Availability.UNKNOWN or self.mem_alloc is None:
            return None
        if not self.schedulable:
            return 0
        usable = self.mem_schedulable if self.mem_schedulable is not None else self.mem_total
        if usable is None or self.mem_completing is None:
            return None
        return max(0, usable - self.mem_alloc - self.mem_completing)

    @property
    def gpus_free(self) -> Optional[int]:
        """Whole GPUs not allocated or shared, on a schedulable node."""
        if self.gpus_total == 0:
            return 0
        if self.availability == Availability.UNKNOWN:
            return None
        if not self.schedulable:
            return 0
        if self.gpus_total is None or self.gpus_alloc is None or self.gpus_shared is None:
            return None
        if self.gpus_completing is None:
            return None
        return max(0, self.gpus_total - self.gpus_alloc - self.gpus_shared - self.gpus_completing)

    @property
    def gpus_stranded_cpu(self) -> Optional[int]:
        """Free GPUs that cannot be started because there is under a core for each."""
        free = self.gpus_free
        if free == 0:
            return 0
        cpus = self.cpus_free
        if free is None or cpus is None:
            return None
        return max(0, free - cpus // (self.threads_per_core or 1))

    @property
    def gpus_stranded_mem(self) -> Optional[int]:
        free = self.gpus_free
        if free == 0:
            return 0
        memory = self.mem_free
        if free is None or memory is None:
            return None
        return free if memory == 0 else 0

    @property
    def gpus_startable(self) -> Optional[int]:
        """Free GPUs less those stranded by exhausted CPUs or tracked memory."""
        free, stranded = self.gpus_free, self.gpus_stranded_cpu
        if free is None or stranded is None:
            return None
        return 0 if self.gpus_stranded_mem else free - stranded

    @property
    def gpus_short_of_default_cpus(self) -> Optional[int]:
        """Free GPUs that cannot also get the partition's DefCpuPerGPU."""
        free, cpus, per_gpu = self.gpus_free, self.cpus_free, self.default_cpus_per_gpu
        if free is None or cpus is None or not per_gpu:
            return None
        return max(0, free - cpus // per_gpu)

    def gpus_free_by_type(self) -> Optional[Dict[str, int]]:
        free = self.gpus_free
        if free is None:
            return None
        if free == 0:
            return {}
        if len(self.gpu_types) == 1:
            return {next(iter(self.gpu_types)): free}
        if self.gpu_types_alloc is None or self.gpus_shared or self.gpus_completing:
            return None
        return {kind: max(0, count - self.gpu_types_alloc.get(kind, 0))
                for kind, count in self.gpu_types.items()}


@dataclass
class Partition:
    name: str
    state: Optional[str] = None  # UP, DOWN, DRAIN, INACTIVE
    is_default: Optional[bool] = None
    hidden: Optional[bool] = None
    nodes: List[str] = field(default_factory=list)
    max_time: Optional[int] = None  # seconds
    max_time_unlimited: bool = False
    default_time: Optional[int] = None
    qos: Optional[str] = None
    allow_accounts: Optional[List[str]] = None  # None = all or unknown
    deny_accounts: Optional[List[str]] = None
    allow_qos: Optional[List[str]] = None
    deny_qos: Optional[List[str]] = None
    allow_groups: Optional[List[str]] = None
    preempt_mode: Optional[str] = None
    priority_tier: Optional[int] = None
    default_cpus_per_gpu: Optional[int] = None
    select_parameters: Optional[str] = None
    tres: Dict[str, float] = field(default_factory=dict)


@dataclass
class Reservation:
    name: str
    state: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    nodes: List[str] = field(default_factory=list)
    partition: Optional[str] = None
    users: List[str] = field(default_factory=list)
    accounts: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    tres: Dict[str, float] = field(default_factory=dict)
