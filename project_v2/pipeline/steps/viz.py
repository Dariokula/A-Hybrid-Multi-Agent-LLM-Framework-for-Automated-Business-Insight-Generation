# pipeline/steps/viz.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

_ALLOWED_SCALES = {"linear", "log", "symlog"}


def _is_categorical_like(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_bool_dtype(s)
        or pd.api.types.is_string_dtype(s)
    )


def _pick_time_x(df: pd.DataFrame, aggregate_plan: Dict[str, Any]) -> Optional[str]:
    if "time_bucket" in df.columns:
        return "time_bucket"
    tc = aggregate_plan.get("time_column")
    if isinstance(tc, str) and tc in df.columns:
        return tc
    for c in df.columns:
        if is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["date", "datum", "time", "zeit", "start", "ende", "timestamp", "monat", "month", "jahr"]):
            return c
    return None


def _pick_numeric_y(df: pd.DataFrame, aggregate_plan: Dict[str, Any], x: Optional[str]) -> Optional[str]:
    return _pick_numeric_y_with_prompt(df, aggregate_plan, x=x, prompt="")


def _is_id_like(col: str) -> bool:
    c = (col or "").lower()
    return any(k in c for k in ["id", "uuid", "guid", "key", "nr", "no", "number", "pos"])


def _score_kpi_candidate(col: str, prompt: str) -> float:
    c = (col or "").lower()
    p = (prompt or "").lower()

    score = 0.0
    if any(k in p for k in ["kpi", "metric", "measure", "value"]):
        score += 0.2

    dev_intent = any(k in p for k in [
        "deviation", "deviations", "abweich", "delay", "delays", "late", "lateness", "overdue",
        "slippage", "gap", "delta", "diff", "difference",
    ])
    if dev_intent:
        if any(k in c for k in [
            "dev", "abweich", "delay", "late", "overdue", "slip", "gap", "delta", "diff",
            "variance", "ta_", "rel", "abwe", "verzug",
        ]):
            score += 2.0
        if any(k in p for k in ["end", "finish", "delivery", "due", "ende"]):
            if any(k in c for k in ["end", "ende", "finish", "delivery", "due"]):
                score += 0.7
        if any(k in p for k in ["start", "begin", "beginn"]):
            if any(k in c for k in ["start", "beginn", "begin"]):
                score += 0.5

    time_intent = any(k in p for k in ["cycle time", "lead time", "throughput", "duration", "dauer", "durchlauf", "dlz"])
    if time_intent and any(k in c for k in ["cycle", "lead", "duration", "dauer", "durchlauf", "dlz", "time"]):
        score += 1.2

    if c in p:
        score += 1.0

    if c in {"n_records", "count", "counts", "n", "rows"}:
        score -= 3.0
    if _is_id_like(c):
        score -= 2.0

    return score


def _pick_numeric_y_with_prompt(
    df: pd.DataFrame,
    aggregate_plan: Dict[str, Any],
    *,
    x: Optional[str],
    prompt: str,
) -> Optional[str]:
    metrics = aggregate_plan.get("metrics") or []
    if isinstance(metrics, list):
        for m in metrics:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            if not isinstance(name, str) or name not in df.columns:
                continue
            if name in {"n_records", "count"}:
                continue
            if is_numeric_dtype(df[name]):
                return name

    best: Optional[Tuple[float, str]] = None
    for c in df.columns:
        if c == x:
            continue
        if not is_numeric_dtype(df[c]):
            continue
        sc = _score_kpi_candidate(str(c), prompt)
        cand = (sc, str(c))
        if best is None or cand[0] > best[0]:
            best = cand

    if best and best[0] > 0.3:
        return best[1]

    for c in df.columns:
        if c == x:
            continue
        if not is_numeric_dtype(df[c]):
            continue
        cl = str(c).lower()
        if cl in {"n_records", "count", "counts", "n", "rows"}:
            continue
        if _is_id_like(cl):
            continue
        return c
    return None


def _pick_group(df: pd.DataFrame, aggregate_plan: Dict[str, Any], x: Optional[str], y: Optional[str]) -> Optional[str]:
    gcols = aggregate_plan.get("groupby_columns") or []
    if isinstance(gcols, list):
        for c in gcols:
            if isinstance(c, str) and c in df.columns and c not in {x, y}:
                s = df[c]
                if _is_categorical_like(s):
                    nun = int(s.nunique(dropna=True))
                    if 1 < nun <= 20:
                        return c

    best: Optional[Tuple[int, str]] = None
    for c in df.columns:
        if c in {x, y}:
            continue
        s = df[c]
        if _is_categorical_like(s):
            nun = int(s.nunique(dropna=True))
            if 1 < nun <= 20:
                cand = (nun, c)
                if best is None or cand[0] < best[0]:
                    best = cand
    return best[1] if best else None


def _pick_categorical_x(df: pd.DataFrame, *, avoid: set[str]) -> Optional[str]:
    best: Optional[Tuple[int, str]] = None
    for c in df.columns:
        if c in avoid:
            continue
        s = df[c]
        if _is_categorical_like(s):
            nun = int(s.nunique(dropna=True))
            if 1 < nun <= 30:
                cand = (nun, c)
                if best is None or cand[0] < best[0]:
                    best = cand
    return best[1] if best else None


def _robust_spread_ratio(y: pd.Series) -> float:
    s = pd.to_numeric(y, errors="coerce").dropna()
    if s.empty:
        return 1.0
    s_pos = s[s > 0]
    if len(s_pos) >= max(5, int(0.5 * len(s))):
        q05 = float(s_pos.quantile(0.05))
        q95 = float(s_pos.quantile(0.95))
        denom = max(q05, 1e-12)
        return q95 / denom
    q05 = float(s.quantile(0.05))
    q95 = float(s.quantile(0.95))
    denom = max(abs(q05), 1e-12)
    return abs(q95) / denom


def _infer_y_scale(
    y: pd.Series,
    *,
    preferred: Optional[str],
    log_ratio_threshold: float,
) -> str:
    if preferred and preferred.lower() in _ALLOWED_SCALES:
        return preferred.lower()

    s = pd.to_numeric(y, errors="coerce").dropna()
    if s.empty:
        return "linear"
    if (s <= 0).any():
        return "linear"

    ratio = _robust_spread_ratio(s)
    return "log" if ratio >= log_ratio_threshold else "linear"


def _iqr_inliers(s: pd.Series, *, k: float = 1.5) -> Tuple[pd.Series, Dict[str, Any]]:
    x = pd.to_numeric(s, errors="coerce")
    xv = x.dropna()
    if xv.empty:
        return x, {"ok": False}
    q1 = float(xv.quantile(0.25))
    q3 = float(xv.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return x, {"ok": True, "low": q1, "high": q3, "iqr": iqr}
    low = q1 - k * iqr
    high = q3 + k * iqr
    inliers = x[(x >= low) & (x <= high)]
    return inliers, {"ok": True, "low": low, "high": high, "iqr": iqr, "q1": q1, "q3": q3}


def _infer_y_limits(
    y: pd.Series,
    *,
    scale: str,
    zero_baseline: str = "auto",
    outlier_policy: Dict[str, Any],
    axis_padding: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    s = pd.to_numeric(y, errors="coerce")
    sv = s.dropna()
    if sv.empty:
        return {"min": None, "max": None}, {"applied": False, "enabled": bool(outlier_policy.get("enabled", True))}

    enabled = bool(outlier_policy.get("enabled", True))
    method = str(outlier_policy.get("method", "iqr")).lower()
    max_frac = float(outlier_policy.get("max_outlier_frac", 0.02))
    min_n = int(outlier_policy.get("min_n", 30))
    k = float(outlier_policy.get("iqr_k", 1.5))

    used = sv
    out_info: Dict[str, Any] = {
        "enabled": enabled,
        "method": method,
        "max_outlier_frac": max_frac,
        "min_n": min_n,
        "iqr_k": k,
        "view_clipped": False,
    }

    if enabled and method == "iqr" and int(sv.shape[0]) >= min_n:
        inliers, stats = _iqr_inliers(s, k=k)
        if stats.get("ok", False):
            out_mask = sv.index.difference(inliers.dropna().index)
            out_n = int(len(out_mask))
            out_frac = out_n / max(1, int(sv.shape[0]))
            out_info.update(
                {"outlier_count": out_n, "outlier_frac": float(out_frac), "fence_low": stats.get("low"), "fence_high": stats.get("high")}
            )
            if out_n > 0 and inliers.dropna().shape[0] >= max(3, int(0.5 * min_n)):
                used = inliers.dropna()
                out_info["view_clipped"] = True

    y_min = float(used.min())
    y_max = float(used.max())

    if scale == "log":
        used_pos = used[used > 0]
        if used_pos.empty:
            return {"min": None, "max": None}, out_info
        y_min = float(used_pos.min())
        y_max = float(used_pos.max())
        return {"min": y_min * 0.9, "max": y_max * 1.1}, out_info

    pad_frac = float(axis_padding.get("pad_frac", 0.03))
    pad_min_rel = float(axis_padding.get("pad_min_rel", 0.01))

    rng = y_max - y_min
    base = max(abs(y_min), abs(y_max), 1.0)
    pad = max(pad_frac * max(rng, 0.0), pad_min_rel * base)

    if zero_baseline == "auto":
        if y_min >= 0:
            return {"min": 0.0, "max": y_max + pad}, out_info
        if y_max <= 0:
            return {"min": y_min - pad, "max": 0.0}, out_info

    return {"min": y_min - pad, "max": y_max + pad}, out_info


def _infer_x_limits(
    x: pd.Series,
    *,
    is_time: bool,
    outlier_policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if bool(is_time):
        return {"min": None, "max": None}, {"enabled": False, "view_clipped": False}
    s = pd.to_numeric(x, errors="coerce")
    sv = s.dropna()
    if sv.empty:
        return {"min": None, "max": None}, {"enabled": bool(outlier_policy.get("enabled", True)), "view_clipped": False}

    enabled = bool(outlier_policy.get("enabled", True))
    method = str(outlier_policy.get("method", "iqr")).lower()
    min_n = int(outlier_policy.get("min_n", 30))
    k = float(outlier_policy.get("iqr_k", 1.5))

    used = sv
    info: Dict[str, Any] = {"enabled": enabled, "method": method, "min_n": min_n, "iqr_k": k, "view_clipped": False}

    if enabled and method == "iqr" and int(sv.shape[0]) >= min_n:
        inliers, stats = _iqr_inliers(s, k=k)
        if stats.get("ok", False):
            out_mask = sv.index.difference(inliers.dropna().index)
            out_n = int(len(out_mask))
            out_frac = out_n / max(1, int(sv.shape[0]))
            info.update({"outlier_count": out_n, "outlier_frac": float(out_frac), "fence_low": stats.get("low"), "fence_high": stats.get("high")})
            if out_n > 0 and inliers.dropna().shape[0] >= max(3, int(0.5 * min_n)):
                used = inliers.dropna()
                info["view_clipped"] = True

    xmin = float(used.min())
    xmax = float(used.max())
    rng = xmax - xmin
    pad = 0.03 * max(rng, 0.0)
    return {"min": xmin - pad, "max": xmax + pad}, info


def _infer_tick_policy(*, is_time_x: bool, n_unique_x: int) -> Dict[str, Any]:
    x_max_ticks = 10 if is_time_x else 12
    if n_unique_x > 0:
        x_max_ticks = min(x_max_ticks, max(3, n_unique_x))
    return {"x": {"max_ticks": int(x_max_ticks), "is_time": bool(is_time_x)}, "y": {"max_ticks": 6}}


def _resolve_aggregate_plan(state) -> Dict[str, Any]:
    agg = state.params.get("aggregate") or {}
    if isinstance(agg, dict) and isinstance(agg.get("plan"), dict):
        return agg["plan"]
    if isinstance(agg, dict):
        return agg
    return {}


# -----------------------------------------------------------------------------
# Domain knowledge enrichment (labels + units)
# -----------------------------------------------------------------------------
def _flatten_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return "\n".join([_flatten_text(v) for v in x.values()])
    if isinstance(x, (list, tuple)):
        return "\n".join([_flatten_text(v) for v in x])
    return str(x)


def _extract_field_descriptions(text: str) -> Dict[str, str]:
    t = (text or "").strip()
    if not t:
        return {}
    out: Dict[str, str] = {}
    for m in re.finditer(r"(?m)^\s*[•\-\*]?\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$", t):
        col = m.group(1).strip()
        desc = m.group(2).strip()
        if col and desc and col not in out:
            out[col] = desc
    return out


def _infer_unit_from_desc(desc: str) -> Optional[str]:
    d = (desc or "").lower()
    m = re.search(r"\(\s*in\s+([a-zA-Z ]+?)\s*\)", d)
    if m:
        u = re.sub(r"\s+", " ", m.group(1).strip())
        return u

    if "day" in d or "tage" in d:
        return "days"
    if "hour" in d or "stunden" in d:
        return "hours"
    if "minute" in d or "min" in d:
        return "minutes"
    if "second" in d or "sek" in d:
        return "seconds"
    return None


def _infer_label_from_desc(desc: str) -> Optional[str]:
    if not desc:
        return None
    s = re.sub(r"\(\s*in\s+.+?\)", "", desc, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*\(.+?\)\s*$", "", s).strip()
    return s if s else None


def _get_dk_text_from_state(state) -> str:
    dk_steps = state.params.get("_domain_knowledge_steps") if hasattr(state, "params") else None
    dk = None
    if isinstance(dk_steps, dict):
        dk = dk_steps.get("viz")
        if dk is None:
            dk = dk_steps.get("columns")
    if dk is None and hasattr(state, "params"):
        dk = state.params.get("_domain_knowledge_latest")
    return _flatten_text(dk)


def _humanize_col(col: str) -> str:
    s = (col or "").strip()
    if not s:
        return ""
    s2 = s.replace("_", " ").strip()
    return s2[:1].upper() + s2[1:]


def _enrich_viz_spec_from_domain_knowledge(viz_spec: Dict[str, Any], *, state, resolved: Dict[str, Any]) -> Dict[str, Any]:
    """
    DK should be the source of truth for axis labels/units when a field definition exists.

    Previous behavior only filled missing labels -> bad LLM labels survived.
    New behavior:
      - If DK has a description for the resolved axis column, we override label and unit.
      - If DK doesn't know the column, keep (or humanize) existing labels.
    """
    vs = dict(viz_spec or {})
    vs.setdefault("units", {})
    if not isinstance(vs.get("units"), dict):
        vs["units"] = {}

    xcfg = vs.get("x") if isinstance(vs.get("x"), dict) else {}
    ycfg = vs.get("y") if isinstance(vs.get("y"), dict) else {}
    vs["x"] = dict(xcfg)
    vs["y"] = dict(ycfg)

    dk_text = _get_dk_text_from_state(state)
    fields = _extract_field_descriptions(dk_text)

    rx = (resolved or {}).get("x")
    ry = (resolved or {}).get("y")

    # Y axis
    if isinstance(ry, str) and ry:
        desc = fields.get(ry)
        if desc:
            lab = _infer_label_from_desc(desc)
            if lab:
                vs["y"]["label"] = lab  # ✅ override if DK knows it
            u = _infer_unit_from_desc(desc)
            if u:
                vs["units"][ry] = u      # ✅ override unit too
        else:
            # DK unknown -> fallback to existing or humanized
            if not (vs["y"].get("label") or "").strip():
                vs["y"]["label"] = _humanize_col(ry)

    # X axis
    if isinstance(rx, str) and rx:
        desc = fields.get(rx)
        if desc:
            lab = _infer_label_from_desc(desc)
            if lab:
                vs["x"]["label"] = lab  # ✅ override if DK knows it
            u = _infer_unit_from_desc(desc)
            if u:
                vs["units"][rx] = u
        else:
            if not (vs["x"].get("label") or "").strip():
                vs["x"]["label"] = _humanize_col(rx)

    return vs


async def step_viz(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents):
    llm_error: Optional[str] = None
    out: Dict[str, Any] = {}
    try:
        out = await agents.run(
            "viz",
            prompt=state.composed_prompt(),
            meta={
                "step": "viz",
                "family": state.params.get("family"),
                "type": state.params.get("type"),
                "filters": state.params.get("filters", []),
                "columns": state.params.get("columns", list(df.columns)),
                "prepare_actions": state.params.get("prepare_actions", []),
                "aggregate": state.params.get("aggregate", {}),
            },
            df_profile=df_profile,
        )
        if not isinstance(out, dict):
            raise ValueError("viz agent returned non-dict output")
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        out = {
            "viz": {
                "title": "Visualization",
                "subtitle": "",
                "x": {"label": "", "unit": None, "date_format": "auto", "tick_rotation": 0},
                "y": {"label": "", "unit": None, "scale": "linear", "format": "auto"},
                "legend": {"show": True, "title": None, "loc": "best", "ncol": 1},
                "palette": {"name": "tableau", "max_colors": 10},
                "style": {"grid": True, "spines": "left_bottom", "alpha": 0.9, "line_width": 2.0},
                "units": {},
            },
            "rationale": "LLM unavailable; using fallback viz spec.",
            "confidence": 0.3,
            "signals": ["llm_failed"],
            "llm_error": llm_error,
        }

    viz_spec = out.get("viz") or {}
    if not isinstance(viz_spec, dict):
        viz_spec = {}

    viz_spec.setdefault("axis_padding", {"pad_frac": 0.03, "pad_min_rel": 0.01})
    viz_spec.setdefault("outlier_policy", {"enabled": True, "method": "iqr", "iqr_k": 1.5, "max_outlier_frac": 0.02, "min_n": 30})

    plan = _resolve_aggregate_plan(state)
    typ = (state.params.get("type") or "").strip().lower()

    x: Optional[str] = None
    y: Optional[str] = None
    group: Optional[str] = None
    is_time_x = False

    if typ == "trend":
        x = _pick_time_x(df, plan)
        y = _pick_numeric_y_with_prompt(df, plan, x=x, prompt=state.composed_prompt())
        group = _pick_group(df, plan, x, y)
        is_time_x = bool(x and x in df.columns and (is_datetime64_any_dtype(df[x]) or x == "time_bucket"))
    elif typ == "group_compare":
        y = _pick_numeric_y_with_prompt(df, plan, x=None, prompt=state.composed_prompt())
        avoid = {c for c in [y, "time_bucket"] if c}
        gb_cols = plan.get("groupby_columns") or []
        cand = None
        if isinstance(gb_cols, list):
            for c in gb_cols:
                if isinstance(c, str) and c in df.columns and c not in avoid and _is_categorical_like(df[c]):
                    nun = int(df[c].nunique(dropna=True))
                    if 1 < nun <= 30:
                        cand = c
                        break
        x = cand or _pick_categorical_x(df, avoid=avoid)
        group = x
        is_time_x = False
    elif typ == "distribution":
        y = _pick_numeric_y_with_prompt(df, plan, x=None, prompt=state.composed_prompt())
        x = _pick_categorical_x(df, avoid=set()) if y is None else y
        group = _pick_group(df, plan, x=None, y=y)
        is_time_x = False
    elif typ == "stats_summary":
        x = None
        y = _pick_numeric_y_with_prompt(df, plan, x=None, prompt=state.composed_prompt())
        group = _pick_group(df, plan, x=None, y=y)
        is_time_x = False
    elif typ == "relationships":
        x = None
        y = None
        group = None
        is_time_x = False
    else:
        x = _pick_time_x(df, plan)
        y = _pick_numeric_y_with_prompt(df, plan, x=x, prompt=state.composed_prompt())
        group = _pick_group(df, plan, x, y)
        is_time_x = bool(x and x in df.columns and (is_datetime64_any_dtype(df[x]) or x == "time_bucket"))

    y_scale_pref = ((viz_spec.get("y") or {}).get("scale") or None) if isinstance(viz_spec.get("y"), dict) else None
    log_ratio_threshold = 1000.0

    y_series = df[y] if (y and y in df.columns and is_numeric_dtype(df[y])) else pd.Series(dtype="float64")
    y_scale = _infer_y_scale(y_series, preferred=y_scale_pref, log_ratio_threshold=log_ratio_threshold)

    y_limits, out_info = _infer_y_limits(
        y_series,
        scale=y_scale,
        zero_baseline="auto",
        outlier_policy=viz_spec.get("outlier_policy") or {},
        axis_padding=viz_spec.get("axis_padding") or {},
    )
    viz_spec["outlier_policy"] = {**(viz_spec.get("outlier_policy") or {}), **out_info}

    x_series = df[x] if (x and x in df.columns and is_numeric_dtype(df[x]) and not bool(is_time_x)) else pd.Series(dtype="float64")
    x_limits, _ = _infer_x_limits(
        x_series,
        is_time=bool(is_time_x),
        outlier_policy=viz_spec.get("outlier_policy") or {},
    )

    n_unique_x = int(df[x].nunique(dropna=True)) if (x and x in df.columns) else 0
    tick_policy = _infer_tick_policy(is_time_x=is_time_x, n_unique_x=n_unique_x)

    axis_policy = {
        "x": {
            **tick_policy["x"],
            "date_format": (viz_spec.get("x") or {}).get("date_format", "auto"),
            "tick_rotation": int((viz_spec.get("x") or {}).get("tick_rotation", 0) or 0),
            "limits": x_limits,
        },
        "y": {
            **tick_policy["y"],
            "scale": y_scale,
            "limits": y_limits,
            "zero_baseline": "auto",
            "log_ratio_threshold": log_ratio_threshold,
        },
    }

    resolved = {"x": x, "y": y, "group": group, "is_time_x": bool(is_time_x)}
    viz_spec["resolved"] = resolved
    viz_spec["axis_policy"] = axis_policy
    viz_spec["group_policy"] = {"group_col": group}

    # ✅ IMPORTANT: now DK overrides bad labels, instead of only filling empty ones
    viz_spec = _enrich_viz_spec_from_domain_knowledge(viz_spec, state=state, resolved=resolved)

    state.params["viz_spec"] = viz_spec
    state.decisions["viz"] = out

    meta = {"decision": out, "rationale": out.get("rationale", ""), "df_delta": None}
    if llm_error:
        meta.setdefault("warnings", []).append(f"viz_llm_error: {llm_error}")
    return meta, df