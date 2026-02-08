# agents/specs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Callable
import re  # required for _parse_expr()

@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    instruction_key: str
    schema_validator: Callable[[Dict[str, Any]], Dict[str, Any]]

def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    raise ValueError("Planner output must be a dict")

def validate_family(d: Dict[str, Any]) -> Dict[str, Any]:
    d = _as_dict(d)
    fam = str(d.get("family", "descriptive")).strip().lower()
    if fam not in {"descriptive", "diagnostic", "predictive", "prescriptive"}:
        fam = "descriptive"

    rationale = str(d.get("rationale", "")).strip()

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    return {"family": fam, "rationale": rationale, "confidence": conf, "signals": signals}

def validate_type(d: Dict[str, Any]) -> Dict[str, Any]:
    d = _as_dict(d)
    typ = str(d.get("type", "stats_summary")).strip().lower()

    rationale = str(d.get("rationale", "")).strip()

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    return {"type": typ, "rationale": rationale, "confidence": conf, "signals": signals}

def validate_filters(d: Dict[str, Any]) -> Dict[str, Any]:
    d = _as_dict(d)

    if "filters" not in d and isinstance(d.get("decision"), dict):
        d = {**d, **(d.get("decision") or {})}

    filt = d.get("filters") or d.get("filter") or []
    if isinstance(filt, dict):
        filt = [filt]
    if not isinstance(filt, list):
        filt = []

    allowed_ops = {"==", "!=", ">", ">=", "<", "<=", "in", "contains"}

    op_aliases = {
        "=": "==",
        "eq": "==",
        "equals": "==",
        "equal": "==",
        "is": "==",
        "==": "==",

        "ne": "!=",
        "neq": "!=",
        "not_equals": "!=",
        "not equals": "!=",
        "!=": "!=",

        "gt": ">",
        ">": ">",

        "gte": ">=",
        "ge": ">=",
        ">=": ">=",

        "lt": "<",
        "<": "<",

        "lte": "<=",
        "le": "<=",
        "<=": "<=",

        "one_of": "in",
        "any_of": "in",
        "in": "in",

        "contains": "contains",
        "includes": "contains",
        "has": "contains",
    }

    def _strip_quotes(x: Any) -> Any:
        if not isinstance(x, str):
            return x
        s = x.strip()
        if (len(s) >= 2) and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
            return s[1:-1].strip()
        return s

    def _parse_expr(expr: str) -> Dict[str, Any] | None:
        if not isinstance(expr, str):
            return None
        e = expr.strip()
        if not e:
            return None

        m = re.match(r"^([A-Za-z0-9_\.\-]+)\s*(==|!=|>=|<=|>|<|in|contains)\s*(.+)$", e, flags=re.IGNORECASE)
        if not m:
            m = re.match(r"^([A-Za-z0-9_\.\-]+)\s*(=)\s*(.+)$", e)
            if not m:
                return None

        col = m.group(1).strip()
        op = m.group(2).strip().lower()
        rhs = m.group(3).strip()

        op = op_aliases.get(op, op)
        if op not in allowed_ops:
            return None

        val: Any = rhs
        if op == "in":
            if (rhs.startswith("[") and rhs.endswith("]")) or (rhs.startswith("(") and rhs.endswith(")")):
                inner = rhs[1:-1].strip()
                if inner:
                    parts = [p.strip() for p in inner.split(",")]
                    val = [_strip_quotes(p) for p in parts if p]
                else:
                    val = []
            else:
                val = [_strip_quotes(rhs)]
        else:
            val = _strip_quotes(rhs)

        return {"column": col, "op": op, "value": val}

    out = []
    for f in filt:
        if isinstance(f, str):
            parsed = _parse_expr(f)
            if parsed:
                out.append(parsed)
            continue

        if not isinstance(f, dict):
            continue

        col = f.get("column")
        if col is None:
            col = f.get("col") or f.get("field") or f.get("name")

        op = f.get("op")
        if op is None:
            op = f.get("operator") or f.get("operation") or f.get("cmp")

        val = f.get("value")
        if val is None and "values" in f:
            val = f.get("values")

        expr = f.get("expr") or f.get("expression") or f.get("filter")
        if (not col or not op) and isinstance(expr, str):
            parsed = _parse_expr(expr)
            if parsed:
                out.append(parsed)
            continue

        if not col or not op:
            continue

        col = str(col).strip()
        op_norm = str(op).strip().lower().replace("  ", " ")
        op_norm = op_aliases.get(op_norm, op_norm)

        if op_norm not in allowed_ops or not col:
            continue

        if op_norm == "in" and val is not None and not isinstance(val, (list, tuple)):
            val = [val]

        if isinstance(val, str):
            val = _strip_quotes(val)
        elif isinstance(val, list):
            val = [_strip_quotes(x) for x in val]

        out.append({"column": col, "op": op_norm, "value": val})

    rationale = str(d.get("rationale", "")).strip()

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    return {"filters": out, "rationale": rationale, "confidence": conf, "signals": signals}

def validate_columns(d: Dict[str, Any]) -> Dict[str, Any]:
    d = _as_dict(d)

    cols = d.get("columns")
    if cols is None:
        cols = d.get("cols")
    if cols is None:
        cols = d.get("selected_columns")

    if cols is None:
        cols = []
    elif isinstance(cols, str):
        cols = [x.strip() for x in cols.split(",") if x.strip()]

    if not isinstance(cols, list):
        cols = []
    cols = [str(c).strip() for c in cols if c is not None and str(c).strip()]

    rationale = str(d.get("rationale", "")).strip()

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    return {"columns": cols, "rationale": rationale, "confidence": conf, "signals": signals}

def validate_prepare(d: Dict[str, Any]) -> Dict[str, Any]:
    d = _as_dict(d)
    actions = d.get("actions") or d.get("prepare_actions") or []
    if not isinstance(actions, list):
        actions = []

    out_actions = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        params = a.get("params", {})
        if not name:
            continue
        if not isinstance(params, dict):
            params = {}
        out_actions.append({"name": str(name).strip(), "params": params})

    rationale = str(d.get("rationale", "")).strip()

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    return {"actions": out_actions, "rationale": rationale, "confidence": conf, "signals": signals}

def validate_aggregate(d: Dict[str, Any]) -> Dict[str, Any]:
    d = _as_dict(d)

    plan_needed = d.get("plan_needed")
    if plan_needed is None:
        plan_needed = bool(d.get("plan") or d.get("aggregate") or False)
    plan_needed = bool(plan_needed)

    plan = d.get("plan") or d.get("aggregate") or {}
    if not isinstance(plan, dict):
        plan = {}

    rationale = str(d.get("rationale", "")).strip()

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    return {"plan_needed": plan_needed, "plan": plan, "rationale": rationale, "confidence": conf, "signals": signals}

def validate_viz(d: Dict[str, Any]) -> Dict[str, Any]:
    d = _as_dict(d)
    kind = str(d.get("kind", "auto")).strip().lower()

    rationale = str(d.get("rationale", "")).strip()

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    params = d.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    return {"kind": kind, "params": params, "rationale": rationale, "confidence": conf, "signals": signals}

def validate_verify(d: Dict[str, Any]) -> Dict[str, Any]:
    d = _as_dict(d)
    status = str(d.get("status", "warn")).strip().lower()
    # Some agents use 'block' instead of 'fail'. Keep it.
    if status not in {"ok", "warn", "fail", "block"}:
        status = "warn"

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    rationale = str(d.get("rationale", "")).strip()

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    return {"status": status, "confidence": conf, "rationale": rationale, "signals": signals}

def validate_finalize(d: Dict[str, Any]) -> Dict[str, Any]:
    """Validate finalize output.

    Small / local models frequently return JSON that is *semantically correct* but does not
    match the exact nested schema (e.g., they omit the top-level 'narrative' key, or use
    'insights'/'caveats'/'next_actions').

    We accept conservative aliases so reporting can reliably render a summary.
    """
    d = _as_dict(d)

    narrative = d.get("narrative")

    # If the model returned a flat payload with summary/insights/etc, treat it as narrative.
    if narrative is None and any(
        k in d for k in ["summary", "highlights", "insights", "warnings", "caveats", "followups", "next_actions"]
    ):
        narrative = dict(d)

    # If it returned just a string, interpret it as the summary.
    if isinstance(narrative, str):
        narrative = {"summary": narrative}

    if not isinstance(narrative, dict):
        narrative = {}

    summary = str(narrative.get("summary", "")).strip()

    highlights = narrative.get("highlights")
    if highlights is None:
        highlights = narrative.get("insights")
    if not isinstance(highlights, list):
        highlights = []
    highlights = [str(x).strip() for x in highlights if str(x).strip()][:8]

    warnings = narrative.get("warnings")
    if warnings is None:
        warnings = narrative.get("caveats")
    if not isinstance(warnings, list):
        warnings = []
    warnings = [str(x).strip() for x in warnings if str(x).strip()][:8]

    followups = narrative.get("followups")
    if followups is None:
        followups = narrative.get("next_actions")
    if not isinstance(followups, list):
        followups = []
    followups = [str(x).strip() for x in followups if str(x).strip()][:8]

    out_narr = {"summary": summary, "highlights": highlights, "warnings": warnings, "followups": followups}

    rationale = str(d.get("rationale", "")).strip()

    conf = d.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    signals = d.get("signals") or []
    if not isinstance(signals, list):
        signals = []
    signals = [str(x) for x in signals][:8]

    return {"narrative": out_narr, "rationale": rationale, "confidence": conf, "signals": signals}

AGENT_SPECS: Dict[str, AgentSpec] = {
    "family": AgentSpec(agent_id="family", instruction_key="family", schema_validator=validate_family),
    "type": AgentSpec(agent_id="type", instruction_key="type", schema_validator=validate_type),
    "filters": AgentSpec(agent_id="filters", instruction_key="filters", schema_validator=validate_filters),
    "columns": AgentSpec(agent_id="columns", instruction_key="columns", schema_validator=validate_columns),
    "prepare": AgentSpec(agent_id="prepare", instruction_key="prepare", schema_validator=validate_prepare),
    "aggregate": AgentSpec(agent_id="aggregate", instruction_key="aggregate", schema_validator=validate_aggregate),
    "viz": AgentSpec(agent_id="viz", instruction_key="viz", schema_validator=validate_viz),
    "verify": AgentSpec(agent_id="verify", instruction_key="verify", schema_validator=validate_verify),
    "finalize": AgentSpec(agent_id="finalize", instruction_key="finalize", schema_validator=validate_finalize),
}