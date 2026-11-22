"""What is happening now.

    collector = LiveCollector()
    snap = collector.snapshot()

    snap.totals.gpus            Resource(total, alloc, free, unavailable, reserved, shared)
    snap.nodes / .jobs          model.Node / model.Job; None means the query failed
    snap.my_jobs, snap.my_roles the invoker's own
    snap.health                 problem nodes, reservations, scheduler, pending by reason, ...
    snap.priority.fairshare_effective    False: fairshare has no effect on priority here
    snap.capabilities, .warnings         what could not be seen, and what looks wrong

    group_by(snap.jobs, "user")          user, account, qos, partition, role, node, reason,
                                         state, name, total, or a callable
    applicable_limits(snap, account, user, qos, partition)
    reconcile_gpus(snap)                 nodes' view against jobs' view

On demand, for a job the viewer opened (none is ever called by snapshot()):

    collector.steps(job)         StepReport   sstat
    collector.log_paths(job)     LogPaths     squeue
    collector.tail_log(job, paths.raw_stdout)  LogTail
    collector.batch_script(job)  Fetched      only on an explicit keypress
    collector.submit_line(job)   Fetched      only on an explicit keypress

Each carries a model.Status; ``denied`` is Slurm's answer and is shown, not
worked around. Job environments are never queried.

Modules: queries (every argv), cluster, nodes, jobs, limits, accounts (parsers),
summary and reconcile (pure derivations), ondemand, logs, collector.
"""

from .collector import LiveCollector
from .limits import applicable_limits
from .logs import LogTail, tail_file
from .ondemand import Fetched, LogPaths, StepReport
from .reconcile import Reconciliation, reconcile_gpus
from .summary import group_by, reason_code

__all__ = ["Fetched", "LiveCollector", "LogPaths", "LogTail", "Reconciliation", "StepReport",
           "applicable_limits", "group_by", "reason_code", "reconcile_gpus", "tail_file"]
