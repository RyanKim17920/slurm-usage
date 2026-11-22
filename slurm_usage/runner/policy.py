"""The allowlist. Pure functions: decide whether an argv is a read-only query.

Everything not listed is refused. In particular:

* scontrol: only ``show <listed entity>`` and ``ping``, at most one -d (a second
  one prints batch scripts on older Slurm), never -u (act as another user).
* sacctmgr: only ``show|list <listed entity>``; RunawayJobs (which can repair)
  and -i are not listed.
* job environments cannot be queried at all.
* a batch script only as exactly ``sacct -j <id> --batch-script`` and a submit
  line only for one job id, so neither can be collected in bulk.

Slurm's option parser accepts options after positionals, clustered short options
and abbreviated long ones. So every dash token is validated wherever it appears,
long options must be spelled in full, and an option's value may not start with
a dash (use ``--sort=-p``). An unknown spelling is refused, never guessed at.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple


class ReadOnlyViolation(ValueError):
    """The caller asked for something that is not an allowlisted read-only query."""


class _Spec(NamedTuple):
    flags: str          # short options without a value
    valued: str         # short options with a value
    long_flags: str
    long_valued: str


_SPECS = {
    "squeue": _Spec(
        "ahlrV", "ALMnoOpqRStuwj",
        "all noheader long array version start me noconvert local hide",
        "account job format Format partition qos reservation sort states user nodelist name "
        "clusters licenses"),
    "sinfo": _Spec(
        "adehFlNrRsTV", "MnoOpSt",
        "all dead exact noheader future long Node responding list-reasons summarize "
        "reservation version noconvert local",
        "nodes format Format partition sort states clusters"),
    "sacct": _Spec(
        "abcDelLnpPTVX", "AEgiIjkKMNoqrRsSuWx",
        "allusers brief completion duplicates helpformat long allclusters noheader parsable "
        "parsable2 truncate version allocations noconvert local array expand-patterns",
        "accounts endtime group gid nnodes ncpus jobs timelimit-min timelimit-max clusters "
        "nodelist format qos partition reason state starttime uid user wckeys associations "
        "delimiter units name"),
    "sreport": _Spec(
        "anpPV", "tT",
        "all_clusters noheader parsable parsable2 version local",
        "tres"),
    "sshare": _Spec(
        "aelmnpPUV", "AMou",
        "all helpformat long partition noheader parsable parsable2 Users version",
        "accounts clusters format users"),
    "sprio": _Spec(
        "hlnwV", "jMopu",
        "noheader long norm weights version local",
        "jobs clusters format partition user"),
    "sstat": _Spec(
        "aenpPV", "jo",
        "allsteps helpformat noheader parsable parsable2 noconvert version",
        "jobs format"),
    "scontrol": _Spec(
        "adoFV", "M",
        "all details oneliner future version local",
        "cluster"),
    "sacctmgr": _Spec(
        "nopPrsV", "",
        "noheader oneliner parsable parsable2 readonly associations version",
        ""),
}  # type: Dict[str, _Spec]

_SCONTROL_SHOW = frozenset(
    "node nodes job jobs step steps partition partitions config assoc_mgr "
    "reservation reservations license licenses hostnames".split())
_SACCTMGR_SHOW = frozenset(
    "account accounts association associations assoc cluster clusters event "
    "events qos tres user users reservation transaction wckey stats".split())
_SREPORT_REPORTS = frozenset(("cluster", "job", "reservation", "user"))


def _split_options(program: str, args: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    spec = _SPECS[program]
    long_flags, long_valued = spec.long_flags.split(), spec.long_valued.split()
    options, words = [], []  # type: List[Tuple[str, str]], List[str]
    index = 0

    def following(name: str) -> str:
        if index >= len(args) or args[index].startswith("-"):
            raise ReadOnlyViolation("%s %s needs a value" % (program, name))
        return args[index]

    while index < len(args):
        token = args[index]
        index += 1
        if not token.startswith("-"):
            words.append(token)
        elif token.startswith("--"):
            name, has_value, value = token[2:].partition("=")
            if name in long_flags and not has_value:
                options.append((name, ""))
            elif name in long_valued:
                if not has_value:
                    value = following(token)
                    index += 1
                options.append((name, value))
            else:
                raise ReadOnlyViolation("%s option %s is not allowed" % (program, token[:40]))
        else:
            letters = token[1:]
            if not letters:
                raise ReadOnlyViolation("bare dash is not allowed")
            for position, letter in enumerate(letters):
                if letter in spec.flags:
                    options.append((letter, ""))
                elif letter in spec.valued:
                    value = letters[position + 1:]
                    if not value:
                        value = following("-" + letter)
                        index += 1
                    options.append((letter, value))
                    break
                else:
                    raise ReadOnlyViolation("%s option -%s is not allowed" % (program, letter))
    return options, words


def is_batch_script_query(argv: Sequence[str]) -> bool:
    return (len(argv) == 4 and list(argv[:2]) == ["sacct", "-j"] and argv[2].isascii()
            and argv[2].isdigit() and argv[3] == "--batch-script")


def check(argv: Sequence[str]) -> None:
    """Raise ReadOnlyViolation unless argv is allowed."""
    if isinstance(argv, (str, bytes)) or not argv or not all(isinstance(a, str) for a in argv):
        raise ReadOnlyViolation("argv must be a list of strings")
    if any("\0" in arg for arg in argv):
        raise ReadOnlyViolation("NUL in argument")
    program = argv[0]
    if program not in _SPECS:
        raise ReadOnlyViolation("%r is not an allowed program" % (program[:40],))
    if is_batch_script_query(argv):
        return
    options, words = _split_options(program, argv[1:])
    names = [name for name, _ in options]
    version_only = not words and bool(names) and set(names) <= {"V", "version"}
    if version_only:
        return

    if program == "scontrol":
        if names.count("d") + names.count("details") > 1:
            raise ReadOnlyViolation("repeated -d can print batch scripts")
        if words == ["ping"]:
            return
        if len(words) < 2 or words[0] != "show" or words[1] not in _SCONTROL_SHOW:
            raise ReadOnlyViolation("only scontrol show and scontrol ping are allowed")
        if words[1] == "hostnames" and (len(words) < 3 or any("/" in word for word in words[2:])):
            # a path makes scontrol read that file; no argument reads the environment
            raise ReadOnlyViolation("scontrol show hostnames needs a hostlist")
    elif program == "sacctmgr":
        if (len(words) < 2 or words[0].lower() not in ("show", "list")
                or words[1].lower() not in _SACCTMGR_SHOW):
            raise ReadOnlyViolation("only sacctmgr show of a listed entity is allowed")
    elif program == "sreport":
        if len(words) < 2 or words[0].lower() not in _SREPORT_REPORTS:
            raise ReadOnlyViolation("only sreport reports are allowed")
    elif words:
        # these take no positionals except squeue/sstat job lists, which -j covers
        raise ReadOnlyViolation("unexpected argument to %s" % program)

    if program == "sacct":
        _check_submit_line(options)


def _check_submit_line(options: List[Tuple[str, str]]) -> None:
    fields = [field.split("%")[0].strip().lower()
              for name, value in options if name in ("o", "format")
              for field in value.split(",")]
    if any(field.startswith("submitl") or field == "all" for field in fields):
        jobs = [value for name, value in options if name in ("j", "jobs")]
        if len(jobs) != 1 or not jobs[0].isdigit():
            raise ReadOnlyViolation("a submit line may only be read for one job id")


def describe(argv: Sequence[str], limit: Optional[int] = 120) -> str:
    """The command for display (formats with random delimiters are elided)."""
    shown, skip = [], False
    for arg in argv:
        if skip:
            shown.append("...")
            skip = False
            continue
        shown.append(arg)
        skip = arg in ("-o", "-O", "--format", "--Format")
    text = " ".join(shown)
    return text[:limit] if limit else text
