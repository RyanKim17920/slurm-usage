"""Every Slurm command the live layer uses, as argv."""

from typing import List

from ..parse import Frame

NODES = ["scontrol", "-a", "-d", "-o", "show", "node"]
JOB_DETAIL = ["scontrol", "-a", "-d", "-o", "show", "job"]
PARTITIONS = ["scontrol", "-a", "-o", "show", "partition"]
RESERVATIONS = ["scontrol", "-a", "-o", "show", "reservation"]
ASSOC_MGR = ["scontrol", "show", "assoc_mgr"]
PING = ["scontrol", "ping"]
CONFIG = ["scontrol", "show", "config"]
COORDINATORS = ["sacctmgr", "-n", "-P", "show", "account", "withcoord",
                "format=account,coordinators"]
ACCOUNTING_CAPACITY = ["sacctmgr", "-n", "-P", "show", "cluster", "format=cluster,tres"]
QOS = ["sacctmgr", "-n", "-P", "show", "qos",
       "format=name,priority,maxwall,grptres,maxtrespu,maxtrespa,maxjobspu,maxjobspa,maxsubmitpu"]

# attribute name, squeue -o code. Delimiters are random, so free text is safe anywhere.
QUEUE_FIELDS = (
    ("display", "%i"), ("job_id", "%A"), ("array_job", "%F"), ("array_task", "%K"),
    ("uid", "%U"), ("user", "%u"), ("account", "%a"), ("qos", "%q"), ("partition", "%P"),
    ("state", "%T"), ("priority", "%Q"), ("nice", "%y"), ("submit", "%V"), ("start", "%S"),
    ("end", "%e"), ("limit", "%l"), ("elapsed", "%M"), ("cpus", "%C"), ("node_count", "%D"),
    ("nodes", "%N"), ("batch_host", "%B"), ("reservation", "%v"), ("gres", "%b"),
    ("wckey", "%w"), ("dependency", "%E"), ("reason", "%r"), ("name", "%j"),
    ("comment", "%k"), ("workdir", "%Z"), ("command", "%o"),
)

STEP_FIELDS = ("JobID", "NTasks", "AveCPU", "AveRSS", "MaxRSS", "TRESUsageInAve",
               "TRESUsageInMax", "TRESUsageInTot", "Nodelist")


def queue(frame: Frame) -> List[str]:
    # -r gives an exact task count for pending arrays in the same single RPC
    return ["squeue", "-h", "-a", "-r", "-t", "all", "-o",
            frame.format([code for _, code in QUEUE_FIELDS])]


def associations(cluster: str) -> List[str]:
    return ["sacctmgr", "-n", "-P", "show", "assoc", "cluster=" + cluster,
            "format=account,user,partition,qos,defaultqos"]


def user(name: str) -> List[str]:
    return ["sacctmgr", "-n", "-P", "show", "user", name, "format=user,admin,defaultaccount"]


def steps(job_id: str) -> List[str]:
    return ["sstat", "-a", "-n", "-P", "-j", job_id, "-o", ",".join(STEP_FIELDS)]


def log_paths(job_id: str, frame: Frame) -> List[str]:
    # an -O suffix also lifts that field's 20 character truncation
    fields = "JobID:%s,STDOUT:%s,STDERR:%s" % (frame.separator, frame.separator, frame.end)
    return ["squeue", "-h", "-a", "-j", job_id, "-O", fields]


def submit_line(job_id: str, delimiter: str) -> List[str]:
    return ["sacct", "-X", "-n", "-P", "--delimiter=" + delimiter, "-j", job_id,
            "-o", "JobIDRaw,SubmitLine"]
