# Data sources

Every command `slurm-usage` runs, with the arguments it passes. All of them go
through `slurm_usage.runner`, which checks the whole argument list against a
read-only allowlist (`slurm_usage/runner/policy.py`) before starting anything,
runs it as you with a timeout and an output cap, and never through a shell.
Field delimiters are random per call, so text in a job name or comment cannot
forge a record.

Slurm decides what each command returns to you. Nothing here checks ownership
or tries to see more than Slurm shows.

## Live view (every refresh: 30 s while in use, backing off to 5 min when idle)

| Command | Used for |
| --- | --- |
| `scontrol -a -d -o show node` | nodes: state, CPUs, memory, GRES/GPUs allocated, reasons |
| `squeue -h -a -r -t all -o <fields>` | the queue: ids, user, account, QOS, partition, state, times, resources, pending reason, job name, comment, working directory, command |
| `scontrol -a -d -o show job` | per-job GPU indices and details squeue does not have; read line by line, each record cut to the part used |
| `scontrol -a -o show partition`, `scontrol -a -o show reservation` | partitions, reservations |
| `scontrol show assoc_mgr` | limits and current usage against them, as the controller sees them |
| `scontrol ping` | whether the controller answers |

## Configuration and accounting tables (cached 10 min)

| Command | Used for |
| --- | --- |
| `scontrol show config` | cluster name, `PrivateData`, `SlurmUser`, accounting settings |
| `sacctmgr -n -P show assoc cluster=<name> format=account,user,partition,qos,defaultqos` | who belongs to which account |
| `sacctmgr -n -P show qos format=name,priority,maxwall,grptres,...` | QOS limits |
| `sacctmgr -n -P show account withcoord format=account,coordinators` | coordinators |
| `sacctmgr -n -P show user <you> format=user,admin,defaultaccount` | your admin level and default account (a display hint only) |
| `sacctmgr -n -P show cluster format=cluster,tres` | the capacity the accounting database records, compared with the nodes |

## History (bulk, cached)

| Command | Used for |
| --- | --- |
| `sacct --helpformat` | which fields this `sacct` supports |
| `sacct -a -X -n -P -S now-1hours -E now -o UID`, read until the first other user | how much history is visible to you |
| `sacctmgr -n -P show event event=cluster start=1970-01-02 format=start`, `sacctmgr -n -P show transaction "action=Add Clusters" format=time` | when accounting began, so the first load knows where to stop |
| `sacct -a -D -X -n -P --delimiter=<random> -S <t1> -E <t2> -o <allocation fields>` | allocations, read newest first in adaptive time slices (a 1 s rest between background slices), then only new records; the last day again at most hourly |
| `sacct -a -D -X -n -P --delimiter=<random> -j <id,id,...> -o <allocation fields>` | re-reading jobs that had not finished, up to 400 ids per call |
| `squeue -a -h -t all -o %A` | which unfinished records are really still running |

Allocation fields: `DBIndex JobIDRaw Cluster JobID Submit Eligible Start End
Suspended ElapsedRaw State AllocTRES ReqTRES ReqMem NNodes AllocCPUS
TimelimitRaw Restarts ExitCode UID User Account QOS Partition NodeList JobName`.

Never requested in bulk and never cached: `SubmitLine`, `Comment`,
`AdminComment`, `WorkDir`, `StdIn`. Job environments are never requested at all.

`-D` (duplicates) is what makes requeued and preempted attempts visible as
separate attempts; without it `sacct` shows only the last one.

## Efficiency (only after pressing `E`)

| Command | Used for |
| --- | --- |
| `sacct -a -D -n -P --delimiter=<random> -S <t1> -E <t2> -o DBIndex,JobIDRaw,Cluster,Submit,End,State,TotalCPU,TRESUsageInTot` (plus a user or account filter when you are on one) | CPU time, peak memory and sampled GPU utilization per step, reduced to three numbers per attempt |

Step queries cost far more than allocation queries, so they run only on request
and only for finished attempts that have not been read before.

## One job, when you open it

| Command | When |
| --- | --- |
| `sacct -a -D -n -P --delimiter=<random> -j <id> -o <detail fields>` | Job detail: every attempt and step |
| `sstat -a -n -P -j <id> -o JobID,NTasks,AveCPU,AveRSS,MaxRSS,TRESUsageInAve,TRESUsageInMax,TRESUsageInTot,Nodelist` | Job detail of a running job |
| `squeue -h -a -j <id> -O JobID,STDOUT,STDERR` | log paths of a job still known to the controller |
| `sacct -j <id> --batch-script` | only when you press `b`; never cached |
| `sacct -a -D -X -n -P --delimiter=<random> -j <id> -o DBIndex,JobIDRaw,Submit,SubmitLine` | only when you press `u`; never cached |
| the log file itself (last 64 KiB) | only when you press `l` or `e`, with your own file permissions |

What `sacct --batch-script` answers, and how it is shown:

| Output | Shown as |
| --- | --- |
| `Batch Script for <id>`, a dashed line, then the script | the script |
| the same header followed by `NONE` | `not stored`: an `srun`/`salloc` job, or the cluster does not set `AccountingStoreFlags=job_script` |
| nothing at all (exit 0) | `denied`: Slurm returned no record to you (scripts normally go only to the owner and to operators), or the job is no longer in accounting |
| an error on stderr | `denied`, `n/a` or `ERR`, from the message |

File paths (working directory, logs, the script named on the submit line) are
otherwise only checked with `os.stat` to say whether they exist and changed
since the job started; they are never opened.
