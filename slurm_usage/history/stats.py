"""Totals for one selection or one group."""

from heapq import heappop, heappush
from typing import NamedTuple, Optional, Tuple

METRICS = ("gpu", "cpu", "mem")
_GIB = 1024 ** 3


class Efficiency(NamedTuple):
    """value: cpu = CPU time over allocated; mem = peak RSS over requested;
    gpu = peak sampled utilization of the allocated GPUs (not a time average).
    coverage: the part of the window's allocated amount the value describes.
    """

    value: Optional[float]
    coverage: Optional[float]


class Stats:
    """Integer totals over the clipped window.

    Additive over every grouping: gpu_seconds, cpu_seconds, mem_byte_seconds.
    Also additive except by node, where a job counts once on each of its nodes:
    attempts, seconds, pending, never_started. Not additive: jobs, peak_gpus,
    active_seconds, first and last.
    """

    __slots__ = (
        "gpu_seconds", "cpu_seconds", "mem_byte_seconds",
        "seconds",  # allocation seconds, summed over attempts
        "active_seconds", "gpu_active_seconds",  # wall time with at least one allocation
        "attempts", "requeue_extra", "first", "last", "peak_gpus",
        "running", "pending", "never_started", "unresolved", "suspension_estimated",
        "gpu_unknown", "cpu_unknown", "mem_unknown",  # attempts whose size is unknown
        "gpu_known", "cpu_known", "mem_known",
        "wait_count", "wait_total", "wait_max",
        "limit_count", "limit_used", "limit_requested",
        "cpu_used", "cpu_covered", "mem_used", "mem_covered", "gpu_used", "gpu_covered",
        "gpu_over_100",
        "gpu_tracked",
        "_cover", "_gpu_cover", "_held", "_ends", "_repeats",
    )

    def __init__(self, gpu_tracked: Optional[bool] = None) -> None:
        self.gpu_seconds = self.cpu_seconds = self.mem_byte_seconds = 0
        self.seconds = self.active_seconds = self.gpu_active_seconds = 0
        self.attempts = self.requeue_extra = self.peak_gpus = 0
        self.first = self.last = None  # type: Optional[int]
        self.running = self.pending = self.never_started = 0
        self.unresolved = self.suspension_estimated = 0
        self.gpu_unknown = self.cpu_unknown = self.mem_unknown = 0
        self.gpu_known = self.cpu_known = self.mem_known = 0
        self.wait_count = self.wait_total = self.limit_count = 0
        self.wait_max = None  # type: Optional[int]
        self.limit_used = self.limit_requested = 0
        self.cpu_used = self.mem_used = self.gpu_used = 0.0
        self.cpu_covered = self.mem_covered = self.gpu_covered = 0
        self.gpu_over_100 = 0
        self.gpu_tracked = gpu_tracked
        self._cover = self._gpu_cover = None  # type: Optional[int]
        self._held = 0
        self._ends = []  # type: list
        self._repeats = None  # type: Optional[set]

    def add(self, p: int, q: int, seconds: int, cpus: Optional[int], gpus: Optional[int],
            mem: Optional[int]) -> None:
        """One attempt. Calls must come in order of p."""
        self.attempts += 1
        self.seconds += seconds
        if cpus is None:
            self.cpu_unknown += 1
        else:
            self.cpu_known += 1
            self.cpu_seconds += cpus * seconds
        if mem is None:
            self.mem_unknown += 1
        else:
            self.mem_known += 1
            self.mem_byte_seconds += mem * seconds
        if gpus is None:
            self.gpu_unknown += 1
        else:
            self.gpu_known += 1
        if q <= p:
            return
        if self._cover is None:
            self.first, self.active_seconds, self._cover = p, q - p, q
        elif q > self._cover:
            self.active_seconds += q - (p if p > self._cover else self._cover)
            self._cover = q
        if not gpus:
            return
        self.gpu_seconds += gpus * seconds
        cover = self._gpu_cover
        if cover is None or q > cover:
            self.gpu_active_seconds += q - (p if cover is None or p > cover else cover)
            self._gpu_cover = q
        ends = self._ends
        while ends and ends[0][0] <= p:
            self._held -= heappop(ends)[1]
        heappush(ends, (q, gpus))
        self._held += gpus
        if self._held > self.peak_gpus:
            self.peak_gpus = self._held

    def note_repeat(self, identity: Tuple[int, Optional[int]]) -> None:
        """Called only for job ids that have several attempts."""
        if self._repeats is None:
            self._repeats = set()
        if identity in self._repeats:
            self.requeue_extra += 1
        else:
            self._repeats.add(identity)

    def finish(self) -> None:
        self.last = self._cover
        self._ends, self._repeats, self._held = [], None, 0

    @property
    def jobs(self) -> int:
        """Distinct jobs. An array task is a job; a requeue is another attempt of one."""
        return self.attempts - self.requeue_extra

    def amount(self, metric: str) -> Optional[int]:
        """gpu-seconds, cpu-seconds or byte-seconds; None when unknown rather than zero."""
        if metric not in METRICS:
            raise ValueError("unknown metric: %s" % metric)
        if metric == "gpu" and self.gpu_tracked is False:
            return None
        if getattr(self, metric + "_unknown") and not getattr(self, metric + "_known"):
            return None
        if metric == "mem":
            return self.mem_byte_seconds
        return self.gpu_seconds if metric == "gpu" else self.cpu_seconds

    def complete(self, metric: str) -> bool:
        """False if some attempts' size is unknown, making the amount a lower bound."""
        return self.amount(metric) is not None and not getattr(self, metric + "_unknown")

    @property
    def gpu_hours(self) -> Optional[float]:
        value = self.amount("gpu")
        return None if value is None else value / 3600.0

    @property
    def cpu_hours(self) -> Optional[float]:
        value = self.amount("cpu")
        return None if value is None else value / 3600.0

    @property
    def mem_gib_hours(self) -> Optional[float]:
        """Requested or allocated memory, not used memory."""
        value = self.amount("mem")
        return None if value is None else value / (3600.0 * _GIB)

    @property
    def mean_wait(self) -> Optional[float]:
        return self.wait_total / self.wait_count if self.wait_count else None

    @property
    def time_limit_used(self) -> Optional[float]:
        """Elapsed over requested, for completed and timed-out attempts that ended here."""
        return self.limit_used / self.limit_requested if self.limit_requested else None

    def efficiency(self, metric: str) -> Efficiency:
        total = self.amount(metric)
        covered = getattr(self, metric + "_covered")
        return Efficiency(getattr(self, metric + "_used") / covered if covered else None,
                          covered / total if total else None)
