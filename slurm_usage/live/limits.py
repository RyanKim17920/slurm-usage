"""``scontrol show assoc_mgr``: roles, limits and current usage.

A limit row with no limit and no usage is omitted, so absence means "nothing
set, nothing in use" wherever this source was available.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .. import parse
from ..model import Limit, Qos, Role, Snapshot
from ..sanitize import clean, clean_id

_LIMIT = re.compile(r"((?:Grp|Max|MinTRES)[A-Za-z]*)=([^\s]*)")
_VALUE = re.compile(r"([^()]*)\(([^()]*)\)")
_NAMED = re.compile(r"(.*)\((\d+)\)")
_YEAR = 366 * 86400


@dataclass
class AssocUser:
    name: str
    uid: Optional[int] = None
    default_account: Optional[str] = None
    admin_level: Optional[str] = None


@dataclass
class AssocData:
    users: List[AssocUser] = field(default_factory=list)
    roles: List[Role] = field(default_factory=list)
    qos: List[Qos] = field(default_factory=list)
    limits: List[Limit] = field(default_factory=list)


def named(text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """"alice(1001)" to ("alice", 1001); an empty name to None."""
    value = (text or "").strip()
    match = _NAMED.fullmatch(value)
    name, number = match.groups() if match else (value, None)
    return clean_id(parse.optional_text(name)), parse.parse_int(number)


def _number(text: str, name: str, metric: Optional[str]) -> Optional[float]:
    if metric is not None:
        if metric in ("mem", "vmem"):
            value = parse.parse_memory(text)  # MiB unless suffixed
        elif metric in ("fs/disk", "gres/gpumem"):
            value = parse.parse_memory(text, "")
        else:
            value = parse.parse_count(text)
        # TRES-minutes become TRES-seconds
        return value * 60 if value is not None and "Mins" in name else value
    plain = parse.parse_float(text)
    if "Wall" in name:
        return plain * 60 if plain is not None else parse.parse_duration(text)
    return plain


def _impossible(used: float, name: str, metric: Optional[str],
                capacity: Dict[str, float]) -> bool:
    if used < 0 or used >= 2.0 ** 63 * 60:
        return True
    if "RunMins" not in name or not metric:
        return False
    if capacity.get(metric):
        # more than a year of the entire cluster still to run: a wrapped or infinite sum
        return used > capacity[metric] * _YEAR
    countable = metric in ("cpu", "node", "billing") or metric.startswith("gres/gpu:")
    return (countable or metric == "gres/gpu") and used >= 2.0 ** 31 * 60


def parse_limit(name: str, value: str, scope: str, capacity: Dict[str, float],
                **owner: Optional[str]) -> List[Limit]:
    if "TRES" in name:
        items = [(key, raw) for key, raw in parse.split_tres(value)]  # type: list
    else:
        items = [(None, value)]
    result = []
    for metric, raw in items:
        limit = Limit(scope=scope, name=name, metric=clean(metric) if metric else None,
                      raw=clean(raw), source="assoc_mgr", **owner)
        pair = _VALUE.fullmatch(raw)
        limit_text, used_text = pair.groups() if pair else (raw, None)
        if limit_text.strip() in ("N", "") or parse.is_unlimited(limit_text):
            limit.unlimited = True
        else:
            limit.limit = _number(limit_text, name, metric)
        if used_text is not None:
            used = _number(used_text, name, metric)
            if used is not None and _impossible(used, name, metric, capacity):
                limit.suspect = True
            else:
                limit.used = used
        if limit.unlimited and not limit.used and not limit.suspect:
            continue
        result.append(limit)
    return result


def parse_assoc_mgr(text: str, capacity: Optional[Dict[str, float]] = None) -> AssocData:
    data, capacity = AssocData(), capacity or {}
    section = None
    role, qos = None, None  # type: Optional[Role], Optional[Qos]
    subscope, subject = None, None  # type: Optional[str], Optional[str]
    parents = {}  # type: Dict[str, str]
    default_accounts = {}  # type: Dict[str, Optional[str]]

    for line in text.split("\n"):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if not stripped:
            continue
        if indent == 0 and stripped in ("User Records", "Association Records", "QOS Records"):
            section, role, qos, subscope, subject = stripped, None, None, None, None
            continue
        if section == "User Records":
            if indent == 0 and stripped.startswith("UserName="):
                kv = parse.key_values(stripped)
                name, uid = named(kv.get("UserName"))
                if name:
                    default = clean_id(parse.optional_text(kv.get("DefAccount")))
                    level = clean(kv.get("AdminLevel")) or None
                    if level and level.lower() == "not set":
                        level = None
                    data.users.append(AssocUser(name, uid, default, level))
                    default_accounts[name] = default
        elif section == "Association Records":
            if indent == 0:
                if not (stripped.startswith("ClusterName=") and " Account=" in stripped):
                    continue
                kv = parse.key_values(stripped)
                account = clean_id(parse.optional_text(kv.get("Account")))
                role = None
                if not account:
                    continue
                user, uid = named(kv.get("UserName"))
                role = Role(account=account, user=user, uid=uid if user else None,
                            partition=clean_id(parse.optional_text(kv.get("Partition"))))
                data.roles.append(role)
            elif role is not None:
                _role_line(role, stripped, capacity, parents)
        elif section == "QOS Records":
            if indent == 0:
                if stripped.startswith("QOS="):
                    name = named(stripped[4:])[0]
                    qos = Qos(name) if name else None
                    subscope = subject = None
                    if qos:
                        data.qos.append(qos)
            elif qos is not None:
                if stripped in ("Account Limits", "User Limits"):
                    subscope = "qos_account" if stripped.startswith("Account") else "qos_user"
                    subject = None
                elif subscope and "=" not in stripped:
                    if stripped in ("No Accounts", "No Users"):
                        subject = None
                    elif subscope == "qos_user":
                        subject = named(stripped)[0]
                    else:
                        subject = clean_id(stripped)
                elif subscope:
                    if subject:
                        _qos_limits(qos, stripped, capacity, subscope, subject)
                else:
                    _qos_line(qos, stripped, capacity)

    for role in data.roles:
        if not role.lineage:
            chain, current = [], role.account  # type: List[str], Optional[str]
            while current and current not in chain:
                chain.append(current)
                current = parents.get(current)
            role.lineage = chain[::-1]
        if role.is_default is None and role.user in default_accounts:
            role.is_default = default_accounts[role.user] == role.account
        data.limits.extend(role.limits)
    for qos in data.qos:
        data.limits.extend(qos.limits)
    return data


def _role_line(role: Role, line: str, capacity: Dict[str, float],
               parents: Dict[str, str]) -> None:
    key, _, value = line.partition("=")
    if key.startswith("Comment"):
        return
    if key.startswith("SharesRaw"):
        named_values = dict(zip(key.split("/"), value.split("/")))
        role.shares_raw = parse.parse_int(named_values.get("SharesRaw"))
        role.shares_norm = parse.parse_float(named_values.get("Norm"))
        role.fairshare_factor = parse.parse_float(named_values.get("Factor"))
    elif key.startswith("UsageRaw"):
        named_values = dict(zip(key.split("/"), value.split("/")))
        role.usage_raw = parse.parse_float(named_values.get("UsageRaw"))
        role.usage_effective = parse.parse_float(named_values.get("Efctv"))
    elif key == "ParentAccount":
        kv = parse.key_values(line)
        parent = parse.optional_text(kv.get("ParentAccount", "").split("(")[0])
        if parent and not role.user:
            parents[role.account] = clean_id(parent) or ""
        path = [part for part in kv.get("Lineage", "").split("/")
                if part and not part.startswith("0-")]
        if path:
            names = [clean_id(part) or "" for part in path]
            # a user's path can continue with its partition
            if role.account in names:
                role.lineage = names[: names.index(role.account) + 1]
        if "DefAssoc" in kv and role.user:
            role.is_default = kv["DefAssoc"] == "Yes"
    else:
        for name, raw in _LIMIT.findall(line):
            role.limits.extend(parse_limit(
                name, raw, "assoc", capacity, account=role.account, user=role.user,
                partition=role.partition))


def _qos_line(qos: Qos, line: str, capacity: Dict[str, float]) -> None:
    key, _, value = line.partition("=")
    if key == "UsageRaw":
        qos.usage_raw = parse.parse_float(value)
    elif key == "Priority":
        qos.priority = parse.parse_int(value)
    elif key == "PreemptMode":
        qos.preempt_mode = clean(value) or None
    elif key != "Comment":
        for name, raw in _LIMIT.findall(line):
            qos.limits.extend(parse_limit(name, raw, "qos", capacity, qos=qos.name))


def _qos_limits(qos: Qos, line: str, capacity: Dict[str, float], scope: str,
                subject: str) -> None:
    for name, raw in _LIMIT.findall(line):
        if scope == "qos_user":
            found = parse_limit(name, raw, scope, capacity, qos=qos.name, user=subject)
        else:
            found = parse_limit(name, raw, scope, capacity, qos=qos.name, account=subject)
        qos.limits.extend(found)


def applicable_limits(snapshot: Snapshot, account: Optional[str], user: Optional[str],
                      qos: Optional[str] = None,
                      partition: Optional[str] = None) -> Optional[List[Limit]]:
    """Everything that can block such a job: its association, the accounts above
    it, the QOS, the QOS per account and per user, and the partition's QOS.

    A configured per-user/per-account QOS limit with no usage record yet is
    returned as a template (account and user None, used None).
    """
    if snapshot.limits is None:
        return None
    chain = {account} if account else set()
    for role in snapshot.roles or ():
        if role.account == account and role.user == user:
            chain.update(role.lineage)
    qos_names = {qos} if qos else set()
    for part in snapshot.partitions or ():
        if part.name == partition and part.qos:
            qos_names.add(part.qos)

    selected = []
    for limit in snapshot.limits:
        if limit.scope == "assoc":
            if limit.user is None:
                keep = limit.account in chain
            else:
                keep = (limit.user == user and limit.account == account
                        and limit.partition in (None, partition))
        else:
            keep = limit.qos in qos_names
            if limit.scope == "qos_user" and limit.user is not None:
                keep = keep and limit.user == user
            elif limit.scope == "qos_account" and limit.account is not None:
                keep = keep and limit.account == account
        if keep:
            selected.append(limit)
    specific = {(x.scope, x.qos, x.name, x.metric) for x in selected if x.user or x.account}
    return [x for x in selected
            if x.user or x.account or x.source == "assoc_mgr"
            or (x.scope, x.qos, x.name, x.metric) not in specific]
