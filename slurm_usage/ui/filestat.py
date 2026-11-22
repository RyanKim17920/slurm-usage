"""Does a path Slurm recorded still exist, and did it change after the job started?

Slurm records paths (working directory, stdout, stderr, the script named on
the submit line), never their contents. This only calls ``os.stat`` with the
viewer's own permissions: nothing is opened or read, pseudo-filesystems are
skipped, and only regular files (or a directory, for a working directory) count.
"""

import os
import shlex
import stat
from typing import NamedTuple, Optional

from ..sanitize import clean

_PSEUDO = ("/proc", "/sys", "/dev")


class PathState(NamedTuple):
    exists: Optional[bool]  # None = could not tell
    changed: Optional[bool]  # modified after the job started
    note: str
    style: str  # canvas style name


def check(raw: Optional[str], started: Optional[float], directory: bool = False) -> PathState:
    if not raw or "\0" in raw or not os.path.isabs(raw):
        return PathState(None, None, "not an absolute path", "dim")
    if any(raw == root or raw.startswith(root + "/") for root in _PSEUDO):
        return PathState(None, None, "system path, not checked", "dim")
    try:
        info = os.stat(raw)
    except FileNotFoundError:
        return PathState(False, None, "gone (from this host)", "bad")
    except PermissionError:
        return PathState(None, None, "cannot check: no permission", "dim")
    except (OSError, ValueError) as error:
        return PathState(None, None, "cannot check: %s" % clean(getattr(error, "strerror", error)),
                         "dim")
    kind_ok = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not kind_ok:
        return PathState(True, None, "exists, not a %s" % ("directory" if directory
                                                            else "regular file"), "warn")
    if directory:
        return PathState(True, None, "exists", "good")
    if started is not None and info.st_mtime > started:
        return PathState(True, True, "exists · changed since job ran", "warn")
    if started is None:
        return PathState(True, None, "exists", "good")
    return PathState(True, False, "exists · unchanged since job started", "good")


def script_from_submit_line(line: str, workdir: Optional[str]) -> Optional[str]:
    """The batch script named on an sbatch line, if one can be picked out."""
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()
    if not tokens or os.path.basename(tokens[0]) != "sbatch":
        return None
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        if token.endswith((".sh", ".sbatch", ".slurm", ".bash", ".job", ".batch")):
            if not os.path.isabs(token) and workdir:
                token = workdir.rstrip("/") + "/" + token
            return token
    return None
