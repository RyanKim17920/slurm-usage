"""``--once`` and ``--json``: the same figures as the Overview, without curses."""

import json
import time
from typing import Any, Dict, List, Optional

from ..live import applicable_limits, group_by  # noqa: F401  (re-exported for callers)
from ..model import Status
from ..sanitize import clean
from . import data as hd
from . import fmt


def _resource(res: Any) -> Dict[str, Any]:
    return {name: getattr(res, name) for name in ("total", "alloc", "completing", "shared",
                                                   "free", "unavailable", "reserved")}


def _round(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(value, digits)


MIN_COVER = 0.05  # as screens.MIN_COVER: below it a figure describes too little


def _eff_text(eff: Any) -> str:
    """'93%·62%': the value with the coverage it describes, always both; '?' when
    nothing or too little (under 5%) of the allocation was measured."""
    if not eff:
        return fmt.NA
    cover = eff["coverage"]
    if eff["value"] is None:
        return fmt.NOT_YET if cover == 0 else fmt.NOT_APPLICABLE
    if cover is not None and cover < MIN_COVER:
        return fmt.NOT_YET
    return fmt.pct(eff["value"]) + ("·" + fmt.pct(cover) if cover is not None else "")


def _eff(eff: Any) -> Optional[Dict[str, Optional[float]]]:
    if eff is None:
        return None
    return {"value": _round(eff[0], 3), "coverage": _round(eff[1], 3)}


def _safe(fn: Any) -> Any:
    try:
        return fn(), None
    except Exception as error:
        return None, {"status": getattr(error, "status", Status.ERROR),
                      "message": clean(getattr(error, "message", None) or error)}


def trend_dict(row: Any) -> Dict[str, Any]:
    return {"range": row.range, "gpu_hours_allocated": _round(row.gpu_hours),
            "cpu_hours_allocated": _round(row.cpu_hours),
            "gpu_utilization_of_live_capacity": _round(row.gpu_util, 4),
            "cpu_utilization_of_live_capacity": _round(row.cpu_util, 4),
            "gpu_util_peak": _eff(row.gpu_eff), "cpu_efficiency": _eff(row.cpu_eff),
            "previous_gpu_hours_allocated": _round(row.gpu_delta),
            "previous_cpu_hours_allocated": _round(row.cpu_delta),
            "attempts": row.attempts, "jobs": row.jobs, "complete": row.complete,
            "previous_complete": row.previous_loaded,
            "series": [int(v) for v in fmt.resample(row.spark or [], 24)]}


def collect(snap: Any, data: Optional[hd.HistoryData], range_name: str,
            user: Optional[str] = None, account: Optional[str] = None,
            history_info: Any = None) -> Dict[str, Any]:
    out = {"generated_at": int(time.time()), "range": range_name}  # type: Dict[str, Any]
    if snap is not None:
        t, h = snap.totals, snap.health
        out["cluster"] = snap.cluster.name
        out["live"] = {
            "taken_at": int(snap.taken_at),
            "gpus": _resource(t.gpus), "cpus": _resource(t.cpus),
            "nodes": t.nodes, "nodes_schedulable": t.nodes_schedulable,
            "node_states": dict(h.node_states),
            "gpus_stranded": None if h.gpus_stranded_cpu is None and h.gpus_stranded_mem is None
            else (h.gpus_stranded_cpu or 0) + (h.gpus_stranded_mem or 0),
            "largest_free_gpu_block": None if h.largest_gpu_block is None else {
                "node": h.largest_gpu_block.node, "gpus": h.largest_gpu_block.gpus},
            "running": sum(1 for j in snap.jobs or () if j.holds_resources),
            "pending": sum(j.task_count for j in snap.jobs or () if j.pending),
            "completing": sum(1 for j in snap.jobs or () if j.state == "COMPLETING"),
            "pending_by_reason": [{"reason": u.key, "pending": u.pending,
                                   "gpus": u.pending_gpus} for u in h.pending_by_reason],
        }
        out["warnings"] = [{"code": w.code, "message": w.message} for w in snap.warnings]
        if data is not None:
            checks, _ = _safe(data.capacity)
            seen = {w.values.get("metric") for w in snap.warnings if w.code == "capacity_mismatch"}
            for check in checks or ():
                if check.mismatch and check.metric not in seen and not any(
                        check.metric in w["message"] for w in out["warnings"]):
                    out["warnings"].append({"code": "capacity_mismatch", "message": (
                        "%s capacity differs: nodes have %d, the accounting database records "
                        "%d; percentages use the nodes" % (check.metric, check.live,
                                                           check.accounting))})
        out["capabilities"] = {name: c.status for name, c in sorted(snap.capabilities.items())}
    gpus = snap.totals.gpus.total if snap else None
    cpus = snap.totals.cpus.total if snap else None

    if data is None:
        out["history"] = {"status": "off"}
        history_ok = False
    else:
        out["history"] = {"status": getattr(history_info, "accounting", None),
                          "visibility": getattr(history_info, "visibility", None),
                          "notes": [n.message for n in getattr(history_info, "notes", ()) or ()]}
        if data.failure:
            out["history"]["failure"] = {"status": data.failure[0], "message": data.failure[1]}
        history_ok = True
    overview = None
    if history_ok and data.has("overview"):
        overview, error = _safe(lambda: data.overview(gpus, cpus))
        if error:
            out["overview_error"] = error
    if history_ok:
        trends, errors = [], []
        summaries = {r.name: r for r in overview.ranges} if overview is not None else {}
        for name in hd.RANGE_NAMES:
            if name in summaries:
                row = data.overview_trend(summaries[name])  # sparkline from the rollup
                if row.spark is None and not hasattr(summaries[name], "series_step") \
                        and name in ("day", "week", "month"):  # longer series cost a full pass
                    spark, _ = _safe(lambda name=name: data.spark(name))
                    row = row._replace(spark=spark)
                trends.append(trend_dict(row))
                continue
            row, error = _safe(lambda name=name: data.trend(name, None, gpus, cpus))
            if row is not None:
                trends.append(trend_dict(row))
            else:
                errors.append(dict(error, range=name))
        out["trend"] = trends
        if errors:
            out["trend_errors"] = errors

    # accounts
    accounts = {}  # type: Dict[str, Dict[str, Any]]
    if snap is not None:
        for role in snap.roles or ():
            entry = accounts.setdefault(role.account, {"account": role.account, "users": set()})
            if role.user:
                entry["users"].add(role.user)
        for usage in group_by([j for j in snap.jobs or () if j.active], "account"):
            entry = accounts.setdefault(usage.key, {"account": usage.key, "users": set()})
            entry.update(running=usage.running, pending=usage.pending, gpus_now=usage.gpus)
    summary = None
    if overview is not None:
        summary = next((r for r in overview.ranges if r.name == range_name), None)
    if summary is not None:
        for name, item in summary.accounts.items():
            entry = accounts.setdefault(name, {"account": name, "users": set()})
            entry["users"].update(item.users)
            entry.update(gpu_hours_allocated=_round(item.amounts.gpu_hours),
                         cpu_hours_allocated=_round(item.amounts.cpu_hours),
                         gpu_share=_round(item.share, 4),
                         gpu_efficiency=_eff(item.amounts.efficiency("gpu")),
                         complete=summary.complete, source="hourly summary",
                         as_of=summary.end)
    elif history_ok:
        report, error = _safe(lambda: data.report(range_name, None, group_by=(("account",),),
                                                  efficiency=True))
        if report is not None:
            total = report.total.gpu_seconds
            for key, stats in report.group("account").items():
                entry = accounts.setdefault(key[0], {"account": key[0], "users": set()})
                entry.update(gpu_hours_allocated=_round(stats.gpu_hours),
                             cpu_hours_allocated=_round(stats.cpu_hours),
                             gpu_share=_round(stats.gpu_seconds / total, 4) if total else None,
                             gpu_efficiency=_eff(stats.efficiency("gpu")))
        elif error:
            out["accounts_error"] = error
    if snap is not None:
        mine = [r.account for r in snap.my_roles or ()]
        for name, entry in accounts.items():
            entry["visibility"] = (hd.visibility(history_info, snap.invoker, name, mine)
                                   if history_ok else "history off")
    rows = []
    for name in sorted(k for k in accounts if k):
        if account and name != account:
            continue
        entry = accounts[name]
        entry["users"] = len(entry["users"])
        rows.append(entry)
    out["accounts"] = rows

    if user and history_ok:
        filt = hd.where(user=user)
        person = {"user": user}  # type: Dict[str, Any]
        trends = []
        for name in hd.RANGE_NAMES:
            row, _ = _safe(lambda name=name: data.trend(name, filt, gpus, cpus))
            if row is not None:
                trends.append(trend_dict(row))
        person["trend"] = trends
        types = {}
        for dim, _ in hd.DIMENSIONS:
            found, error = _safe(lambda dim=dim: data.run_types(range_name, filt, dim))
            if found is None:
                person["error"] = error
                break
            types[dim] = [{"type": k.label, "attempts": k.attempts, "jobs": k.jobs,
                           "gpu_hours_allocated": _round(k.gpu_hours), "share": _round(k.share, 4),
                           "median_runtime": k.runtime, "median_wait": k.wait,
                           "time_limit_use": _round(k.limit_use, 3),
                           "gpu_util_peak": _round(k.gpu_eff, 3),
                           "gpu_util_coverage": _round(k.gpu_cover, 3),
                           "cpu_efficiency": _round(k.cpu_eff, 3),
                           "cpu_coverage": _round(k.cpu_cover, 3),
                           "failure_rate": _round(k.failure, 3), "complete": k.complete}
                          for k in found]
        person["run_types"] = types
        found, error = _safe(lambda: data.events(range_name, filt))
        if found is not None:
            person["events"] = [{"at": e.at, "kind": e.kind, "text": e.text} for e in found]
        elif error:
            person["events_error"] = error
        out["person"] = person
    return out


def render(summary: Dict[str, Any], width: int = 100) -> str:
    lines = []  # type: List[str]
    add = lines.append
    live = summary.get("live")
    add("%s · range %s · %s" % (summary.get("cluster") or "cluster", summary["range"],
                                time.strftime("%Y-%m-%d %H:%M", time.localtime(summary["generated_at"]))))
    if live:
        for label in ("gpus", "cpus"):
            r = live[label]
            add("%-5s %s total  %s alloc  %s completing  %s free  %s unavailable  %s reserved" % (
                label.upper()[:3], *(fmt.num(r[k]) for k in ("total", "alloc", "completing", "free",
                                                              "unavailable", "reserved"))))
        states = live["node_states"]
        down, drain = fmt.down_drain(states)
        block = live["largest_free_gpu_block"]
        add("Nodes %s (%s schedulable)  %d down  %d drain  %d completing jobs  stranded GPUs %s"
            "  largest free block %s" % (fmt.num(live["nodes"]), fmt.num(live["nodes_schedulable"]),
                                         down, drain, live["completing"],
                                         fmt.num(live["gpus_stranded"]),
                                         "%d GPUs on %s" % (block["gpus"], block["node"])
                                         if block else fmt.NOT_APPLICABLE))
        reasons = " · ".join("%s %d" % (r["reason"], r["pending"]) for r in live["pending_by_reason"][:8])
        add("Queue %d running  %d pending%s" % (live["running"], live["pending"],
                                                ": " + reasons if reasons else ""))
    else:
        add("live: unavailable")
    trends = summary.get("trend")
    if trends is not None:
        add("")
        add("%-6s %12s %6s %12s %6s %10s %10s %7s  %s" % ("Range", "GPU-h alloc", "util",
                                                          "CPU-h alloc", "util", "GPU util*",
                                                          "CPU eff", "ΔGPU", "trend"))
        for t in trends:
            mark = "" if t["complete"] else "≥"
            ge, ce = t["gpu_util_peak"], t["cpu_efficiency"]
            add("%-6s %12s %6s %12s %6s %10s %10s %7s  %s" % (
                t["range"], mark + fmt.hours(t["gpu_hours_allocated"]),
                _marked(mark, fmt.pct(t["gpu_utilization_of_live_capacity"])),
                mark + fmt.hours(t["cpu_hours_allocated"]),
                _marked(mark, fmt.pct(t["cpu_utilization_of_live_capacity"])),
                _eff_text(ge), _eff_text(ce),
                fmt.delta(t["gpu_hours_allocated"], t["previous_gpu_hours_allocated"])
                if t["previous_complete"] else fmt.NOT_YET,
                fmt.sparkline(t["series"], 16).strip()))
        add("util = allocated hours ÷ (live capacity × window); ≥ = window only partly loaded;"
            " ? = not loaded, or measured over under 5%; GPU util* = peak sampled per-GPU"
            " utilization·the share of allocated hours measured")
        for error in summary.get("trend_errors", ()):
            add("%s: %s %s" % (error["range"], fmt.placeholder(error["status"]), error["message"]))
    else:
        add("history: %s" % (summary.get("history") or {}).get("status"))
    accounts = summary.get("accounts") or []
    if accounts:
        add("")
        add("%-20s %7s %5s %5s %6s %9s %7s %10s  %s" % ("Account", "members", "run", "pend",
                                                        "GPUs", "GPU-h", "share", "GPU util*",
                                                        "visible"))
        accounts = sorted(accounts, key=lambda a: -(a.get("gpu_hours_allocated") or 0))
        for a in accounts:
            eff = a.get("gpu_efficiency")
            share = a.get("gpu_share")
            add("%-20s %7s %5s %5s %6s %9s %7s %10s  %s" % (
                fmt.fit(a["account"], 20).rstrip(), a["users"], a.get("running", 0),
                a.get("pending", 0), fmt.num(a.get("gpus_now", 0)),
                ("" if a.get("complete", True) else "≥") + fmt.hours(a.get("gpu_hours_allocated", 0)),
                fmt.pct(share, 1) if share else fmt.NOT_APPLICABLE,
                _eff_text(eff) if eff else fmt.NOT_APPLICABLE,
                a.get("visibility", "")))
        source = next((a for a in accounts if a.get("source")), None)
        if source is not None:
            add("accounts: from the hourly summary, to %s; members = users with an association "
                "or usage" % fmt.when(source.get("as_of")))
    person = summary.get("person")
    if person:
        add("")
        add("Person %s" % person["user"])
        for t in person.get("trend", ()):
            mark = "" if t["complete"] else "≥"
            add("  %-6s %10s GPU-h alloc  %10s CPU-h alloc  %s jobs" % (
                t["range"], mark + fmt.hours(t["gpu_hours_allocated"]),
                mark + fmt.hours(t["cpu_hours_allocated"]), mark + str(t["jobs"])))
        for kind in (person.get("run_types") or {}).get("gpu_count", ()):
            util = {"value": kind["gpu_util_peak"], "coverage": kind["gpu_util_coverage"]}
            add("  %-12s %5d attempts %5d jobs %8s GPU-h  GPU util* %s  fail %s" % (
                kind["type"], kind["attempts"], kind["jobs"], fmt.hours(kind["gpu_hours_allocated"]),
                _eff_text(util), fmt.pct(kind["failure_rate"])))
        if person.get("run_types"):
            add("  fail = failed, node_fail or timeout among attempts that ended")
        for event in person.get("events", ())[:10]:
            add("  %s  %s" % (fmt.when(event["at"]), event["text"]))
    for warning in summary.get("warnings", ()):
        add("warning: " + warning["message"])
    history = summary.get("history") or {}
    for note in history.get("notes", ()):
        add("history: " + note)
    return "\n".join(clean(line) for line in lines) + "\n"


def _marked(mark: str, value: str) -> str:
    """A lower bound only if there is a number to bound."""
    return value if value in fmt.PLACEHOLDERS else mark + value


def to_json(summary: Dict[str, Any]) -> str:
    # ASCII escapes: safe on any terminal or pipe encoding, and still valid JSON
    return json.dumps(summary, indent=2, sort_keys=True, default=str, ensure_ascii=True)


_ASCII = {"±": "+/-", "·": "-", "×": "x", "÷": "/", "Δ": "d", "—": "-", "…": "...", "≥": ">=",
          "░": ".", "▒": ":", "▓": "#", "│": "|", "─": "-", "›": ">", "→": "->", "←": "<-"}
_ASCII.update(zip(fmt.SPARKS, fmt.ASCII_SPARKS))
_ASCII.update(dict.fromkeys("▉▊▋▌▍▎▏", "#"))


def to_ascii(text: str) -> str:
    """The same text with ASCII glyphs; anything else left becomes "?"."""
    out = "".join(_ASCII.get(char, char) for char in text)
    return out.encode("ascii", "replace").decode("ascii")


_ONE_CELL = {"…": ".", "±": "+", "→": ">", "←": "<", "≥": ">", "≤": "<", "▼": "v", "▲": "^",
             "┌": "+", "┐": "+", "└": "+", "┘": "+", "├": "+", "┤": "+", "┊": ":", "▏": "|",
             "↵": "<", "▌": "|", "↑": "^", "↓": "v", "‹": "<", "»": ">", "«": "<",
             "╭": "+", "╮": "+", "╰": "+", "╯": "+", "■": "#", "□": ".", "◆": "*",
             "●": "*", "○": "o", "▶": ">"}


def ascii_cell(char: str) -> str:
    """One screen cell in ASCII: the same width, whatever the glyph."""
    if not char or ord(char[0]) < 128:
        return char if char else " "  # "" is the tail of a double-width character
    found = _ONE_CELL.get(char) or _ASCII.get(char, "?")
    return found if len(found) == 1 else found[0]


def can_encode(encoding: Optional[str]) -> bool:
    """Whether a stream in this encoding can show the glyphs render() uses."""
    try:
        "".join(_ASCII).encode(encoding or "ascii")
        return True
    except (LookupError, UnicodeError):
        return False
