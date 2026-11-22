"""sacctmgr tables: associations, QOS, admin/coordinator, recorded capacity."""

from typing import Dict, List, Optional, Tuple

from .. import parse
from ..model import Limit, Qos, Role
from ..sanitize import clean, clean_id


def table(text: str, width: int) -> List[List[str]]:
    rows = [line.split("|") for line in text.split("\n") if line]
    return [row for row in rows if len(row) == width]


def parse_associations(text: str) -> List[Role]:
    roles = []
    for account, user, partition, qos, default_qos in table(text, 5):
        name = clean_id(parse.optional_text(account))
        if name:
            roles.append(Role(
                account=name, user=clean_id(parse.optional_text(user)),
                partition=clean_id(parse.optional_text(partition)),
                qos_allowed=[clean_id(q) or "" for q in qos.split(",") if q],
                default_qos=clean_id(parse.optional_text(default_qos))))
    return roles


def merge_roles(manager: Optional[List[Role]], database: Optional[List[Role]]
                ) -> Optional[List[Role]]:
    """assoc_mgr has usage and fairshare; slurmdbd has the allowed QOS."""
    if manager is None and database is None:
        return None
    merged = list(manager or ())
    by_key = {(r.account, r.user, r.partition): r for r in merged}
    for role in database or ():
        mine = by_key.get((role.account, role.user, role.partition))
        if mine is None:
            merged.append(role)
        else:
            mine.qos_allowed, mine.default_qos = role.qos_allowed, role.default_qos
    return merged


# sacctmgr column, name as assoc_mgr spells it, scope
_QOS_COLUMNS = (
    ("MaxWallPJ", "qos"), ("GrpTRES", "qos"), ("MaxTRESPU", "qos_user"),
    ("MaxTRESPA", "qos_account"), ("MaxJobsPU", "qos_user"), ("MaxJobsPA", "qos_account"),
    ("MaxSubmitJobsPU", "qos_user"),
)


def parse_qos(text: str) -> List[Qos]:
    result = []
    for name, priority, *values in table(text, 2 + len(_QOS_COLUMNS)):
        qos_name = clean_id(parse.optional_text(name))
        if not qos_name:
            continue
        qos = Qos(qos_name, priority=parse.parse_int(priority))
        for (column, scope), raw in zip(_QOS_COLUMNS, values):
            if not raw:
                continue
            items = []  # type: List[Tuple[Optional[str], Optional[float], str]]
            if "TRES" in column:
                numbers = parse.parse_tres(raw)
                items = [(key, numbers.get(key), part) for key, part in parse.split_tres(raw)]
            elif "Wall" in column:
                items = [(None, parse.parse_duration(raw), raw)]
            else:
                items = [(None, parse.parse_float(raw), raw)]
            for metric, value, part in items:
                if value is not None:
                    qos.limits.append(Limit(
                        scope=scope, name=column, metric=clean(metric) if metric else None,
                        limit=value, qos=qos_name, raw=clean(part), source="sacctmgr"))
        result.append(qos)
    return result


def merge_qos(manager: Optional[List[Qos]], database: Optional[List[Qos]]
              ) -> Optional[List[Qos]]:
    """Copies, because both inputs may be cached between snapshots."""
    if manager is None and database is None:
        return None
    merged = {}  # type: Dict[str, Qos]
    for qos in (manager or []) + (database or []):
        mine = merged.setdefault(
            qos.name, Qos(qos.name, qos.priority, qos.preempt_mode, qos.usage_raw))
        if mine.priority is None:
            mine.priority = qos.priority
        known = {(x.scope, x.name, x.metric) for x in mine.limits
                 if x.user is None and x.account is None}
        mine.limits.extend(
            x for x in qos.limits
            if x.source != "sacctmgr" or (x.scope, x.name, x.metric) not in known)
    return list(merged.values())


def parse_user(text: str, user: str) -> Tuple[Optional[str], Optional[str]]:
    """(admin level, default account)."""
    for name, admin, account in table(text, 3):
        if name == user:
            return clean(admin) or None, clean_id(parse.optional_text(account))
    return None, None


def parse_coordinated(text: str, user: str) -> List[str]:
    return sorted({clean_id(account) or "" for account, names in table(text, 2)
                   if account and user in names.split(",")})


def parse_capacity(text: str, cluster: Optional[str]) -> Optional[Dict[str, float]]:
    rows = table(text, 2)
    for name, tres in rows:
        if name == cluster or (cluster is None and len(rows) == 1):
            return parse.parse_tres(tres)
    return None
