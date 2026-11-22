#!/usr/bin/env python3
"""Try slurm-usage without a cluster: run it against the invented cluster "orion".

    python3 scripts/demo/run_demo.py                # the full-screen explorer
    python3 scripts/demo/run_demo.py --once         # any slurm-usage option works
    python3 scripts/demo/run_demo.py --anonymize

The fake Slurm commands in scripts/demo/bin are the only ones on PATH, so a real
cluster is never asked, even on a login node. You appear as "alice". History is
cached in a scratch directory that is removed afterwards (set SLURM_USAGE_CACHE
to a private directory to keep it), and the first start loads about four months of invented accounting.
"""

import os
import pwd
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True  # keep scripts/demo free of __pycache__

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEMO_USER = "alice"


def demo_environment(scratch: str) -> dict:
    """os.environ with only the fake Slurm commands (and python3 for them) on PATH."""
    pybin = os.path.join(scratch, "python")
    os.makedirs(pybin, exist_ok=True)
    link = os.path.join(pybin, "python3")
    if not os.path.exists(link):
        os.symlink(sys.executable, link)
    state = os.path.join(scratch, "state")
    os.makedirs(state, mode=0o700, exist_ok=True)
    files = os.path.join(scratch, "fs")
    os.makedirs(files, exist_ok=True)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SLURM_", "SQUEUE_", "SACCT_", "SINFO_", "SBATCH_"))}
    env["PATH"] = os.pathsep.join((os.path.join(HERE, "bin"), pybin))
    env["SLURM_DEMO_STATE"] = state
    env["SLURM_DEMO_FILES"] = files
    return env


def write_logs(user: str, days: int = 3) -> int:
    """Logs for the demo user's recent jobs, so l / e in Job detail show a tail."""
    sys.path.insert(0, HERE)
    import fakeslurm
    view = fakeslurm.View()
    count = 0
    for attempt in view.attempts:
        if (attempt.user != user or view.status(attempt) == "PENDING"
                or (attempt.end and attempt.end < view.t - days * fakeslurm.DAY)):
            continue
        path = fakeslurm.log_path(attempt)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(fakeslurm.log_text(view, attempt))
        with open(fakeslurm.log_path(attempt, "err"), "w") as handle:
            handle.write("")
        count += 1
    return count


def demo_getpwuid(name: str):
    """pwd.getpwuid, except that this user's own entry carries the demo name."""
    real = pwd.getpwuid
    uid = os.getuid()

    def getpwuid(number: int):
        found = real(number)
        if number != uid:
            return found
        values = list(found)
        values[0], values[5] = name, "/home/" + name
        return pwd.struct_passwd(values)

    return getpwuid


def pretend_to_be(name: str) -> None:
    """The demo's "Me" is alice: only this process's view of its own name changes."""
    pwd.getpwuid = demo_getpwuid(name)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # short and generic: these paths show up on screen
    keep = os.environ.get("SLURM_USAGE_CACHE")
    # short and generic: these paths show up on screen
    scratch = tempfile.mkdtemp(prefix="orion-", dir="/tmp" if os.path.isdir("/tmp") else None)
    try:
        env = demo_environment(scratch)
        os.environ.clear()
        os.environ.update(env)
        if keep:
            os.environ["SLURM_USAGE_CACHE"] = keep
        else:
            cache = os.path.join(scratch, "cache")
            os.makedirs(cache, mode=0o700)
            os.environ["SLURM_USAGE_CACHE"] = cache
        write_logs(DEMO_USER)
        pretend_to_be(DEMO_USER)
        sys.path.insert(0, ROOT)
        from slurm_usage.cli import main as slurm_usage
        return slurm_usage(argv)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
