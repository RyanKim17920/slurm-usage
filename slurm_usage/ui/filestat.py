"""Does a path Slurm recorded still exist, and did it change after the job?

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


GRACE = 120  # seconds: a log is flushed and closed a little after the job's end time


class PathState(NamedTuple):
    exists: Optional[bool]  # None = could not tell
    changed: Optional[bool]  # modified after ``since`` (plus GRACE)
    note: str
    style: str  # canvas style name


def check(raw: Optional[str], since: Optional[float], directory: bool = False,
          running: bool = False, what: str = "the job ended") -> PathState:
    """``since``: when the file should have stopped changing: a log, the job's end (a
    running job's log is simply being written); a script, the job's start. A change
    within GRACE seconds after it is the job's own last write, not a later one."""
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
    if running:
        return PathState(True, None, "exists · being written", "good")
    if since is None:
        return PathState(True, None, "exists", "good")
    if info.st_mtime > since + GRACE:
        return PathState(True, True, "exists · changed after %s" % what, "warn")
    return PathState(True, False, "exists · unchanged since %s" % what, "good")


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
