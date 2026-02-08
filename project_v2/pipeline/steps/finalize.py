# pipeline/steps/finalize.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _compact_trace(state) -> List[Dict[str, Any]]:
    """
    Slim trace: one entry per step with only the core knobs + df_delta.
    This avoids dumping large 'decision' payloads into finalize.
    """
    out: List[Dict[str, Any]] = []
    for h in getattr(state, "history", []) or []:
        step = getattr(h, "step", "?")
        meta = getattr(h, "meta", {}) or {}
        d = getattr(h, "decision", {}) or {}
        df_delta = meta.get("df_delta")

        item: Dict[str, Any] = {"step": step, "df_delta": df_delta}

        if step == "filters":
            item["filters"] = d.get("filters", [])
        elif step == "columns":
            item["columns"] = d.get("columns", [])
        elif step == "prepare":
            item["prepare_actions"] = d.get("actions", []) or d.get("prepare_actions", [])
        elif step == "aggregate":
            plan = (d.get("plan") or {}) if isinstance(d, dict) else {}
            item["aggregate"] = {
                "time_column": plan.get("time_column"),
                "time_granularity": plan.get("time_granularity"),
                "groupby_columns": plan.get("groupby_columns"),
                "metrics": plan.get("metrics"),
            }
        elif step == "viz":
            viz = d.get("viz") if isinstance(d, dict) else None
            if isinstance(viz, dict):
                item["viz"] = {
                    "title": viz.get("title"),
                    "x": (viz.get("x") or {}).get("label") if isinstance(viz.get("x"), dict) else None,
                    "y": (viz.get("y") or {}).get("label") if isinstance(viz.get("y"), dict) else None,
                }
            else:
                item["viz"] = {"title": d.get("title") if isinstance(d, dict) else None}
        elif step == "verify":
            item["verify_status"] = d.get("status") if isinstance(d, dict) else None

        out.append(item)
    return out


def _collect_domain_knowledge_used(state, max_snippets: int = 4) -> List[Dict[str, Any]]:
    """
    Collect snippets that were actually retrieved per step (from step_input.domain_knowledge.snippets).
    Keep it very small.
    """
    used: List[Dict[str, Any]] = []
    seen = set()

    for h in getattr(state, "history", []) or []:
        meta = getattr(h, "meta", {}) or {}
        step_input = meta.get("step_input") or {}
        dk = step_input.get("domain_knowledge") or {}
        snippets = dk.get("snippets") or []
        if not isinstance(snippets, list):
            continue

        for sn in snippets:
            if not isinstance(sn, dict):
                continue
            source = str(sn.get("source", "") or "")
            text = str(sn.get("text", "") or "")
            score = _safe_float(sn.get("score"))
            key = (source, text[:120])
            if key in seen:
                continue
            seen.add(key)
            used.append(
                {
                    "source": source,
                    "text": text[:350],
                    "score": score,
                    "type": sn.get("type"),
                }
            )

    used_sorted = sorted(
        used,
        key=lambda x: (x.get("score") is not None, x.get("score") or 0.0),
        reverse=True,
    )
    return used_sorted[:max_snippets]


def _get_aggregation_cockpit(state) -> Dict[str, Any]:
    cockpit = state.params.get("aggregation_cockpit") if hasattr(state, "params") else None
    if isinstance(cockpit, dict) and cockpit:
        return cockpit

    ag = state.decisions.get("aggregate") if hasattr(state, "decisions") else None
    if isinstance(ag, dict):
        c = ag.get("cockpit")
        if isinstance(c, dict) and c:
            return c

    return {}


def _extract_counts_examples(
    counts_df: Optional[pd.DataFrame],
    group_cols: List[str],
    x_col: str,
    n_col: str = "n_records",
    top_k: int = 3,
) -> Dict[str, Any]:
    if counts_df is None or not isinstance(counts_df, pd.DataFrame) or counts_df.empty:
        return {}

    cols_needed = [c for c in [x_col, n_col] + group_cols if c in counts_df.columns]
    if x_col not in cols_needed or n_col not in cols_needed:
        return {}

    df = counts_df[cols_needed].copy()
    df[n_col] = pd.to_numeric(df[n_col], errors="coerce")

    try:
        df[x_col] = df[x_col].astype(str)
    except Exception:
        df[x_col] = df[x_col].map(lambda v: str(v))

    df_sorted = df.sort_values(n_col, ascending=True).dropna(subset=[n_col])
    lowest = df_sorted.head(top_k)
    highest = df_sorted.tail(top_k).sort_values(n_col, ascending=False)

    return {
        "lowest_n_buckets": lowest.to_dict(orient="records"),
        "highest_n_buckets": highest.to_dict(orient="records"),
    }


def _verify_summary(verify: Dict[str, Any], max_findings: int = 3) -> Dict[str, Any]:
    if not isinstance(verify, dict) or not verify:
        return {"status": None, "confidence": None, "top_findings": [], "user_warning": ""}

    det = verify.get("deterministic") if isinstance(verify.get("deterministic"), dict) else {}
    findings = det.get("findings") if isinstance(det.get("findings"), list) else []

    top: List[Dict[str, Any]] = []
    for f in findings[:max_findings]:
        if not isinstance(f, dict):
            continue
        top.append(
            {
                "code": f.get("code"),
                "severity": f.get("severity"),
                "message": (str(f.get("message") or "")[:220]),
            }
        )

    uw = verify.get("user_warning") or ""
    if isinstance(uw, str):
        uw = uw[:320]
    else:
        uw = ""

    return {
        "status": verify.get("status"),
        "confidence": verify.get("confidence"),
        "top_findings": top,
        "user_warning": uw,
        "llm_error": verify.get("llm_error"),
    }


def _trend_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_cols: List[str],
    unit: Optional[str],
    *,
    extended: bool = False,
) -> Dict[str, Any]:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return {}

    d = df.copy()
    d[x_col] = pd.to_datetime(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    if d.empty:
        return {}

    def one_series_stats(sdf: pd.DataFrame) -> Dict[str, Any]:
        sdf = sdf.sort_values(x_col)
        y = sdf[y_col].astype(float)
        x0 = sdf[x_col].iloc[0]
        x1 = sdf[x_col].iloc[-1]
        base = {
            "n_points": int(len(y)),
            "mean": float(y.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
            "start": float(y.iloc[0]),
            "end": float(y.iloc[-1]),
            "delta_end_start": float(y.iloc[-1] - y.iloc[0]),
            "x_start": x0.isoformat() if hasattr(x0, "isoformat") else str(x0),
            "x_end": x1.isoformat() if hasattr(x1, "isoformat") else str(x1),
            "unit": unit,
            "x_col": x_col,
            "y_col": y_col,
        }
        if extended:
            base.update(
                {
                    "median": float(y.median()),
                    "p05": float(y.quantile(0.05)),
                    "p95": float(y.quantile(0.95)),
                }
            )
        return base

    out: Dict[str, Any] = {"global": one_series_stats(d), "per_group": {}, "group_cols": group_cols}
    if group_cols:
        gb = d.groupby(group_cols, dropna=False)
        for i, (gkey, sdf) in enumerate(gb):
            if i >= 10:
                break
            out["per_group"][str(gkey)] = one_series_stats(sdf)
    return out


def _distribution_stats(df: pd.DataFrame, y_col: str, unit: Optional[str], *, extended: bool = False) -> Dict[str, Any]:
    if df.empty or y_col not in df.columns:
        return {}
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    if y.empty:
        return {}
    base = {
        "n": int(y.shape[0]),
        "mean": float(y.mean()),
        "min": float(y.min()),
        "max": float(y.max()),
        "unit": unit,
        "y_col": y_col,
    }
    if extended:
        base.update(
            {
                "median": float(y.median()),
                "std": float(y.std(ddof=1)) if y.shape[0] > 1 else 0.0,
                "p05": float(y.quantile(0.05)),
                "p95": float(y.quantile(0.95)),
                "p99": float(y.quantile(0.99)),
            }
        )
    return base


def _format_filter(filters: Any) -> str:
    if not isinstance(filters, list) or not filters:
        return "none"
    parts = []
    for f in filters[:4]:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        op = f.get("op")
        val = f.get("value")
        parts.append(f"{col} {op} {val}")
    return " AND ".join(parts) if parts else "none"


def _format_agg(plan: Dict[str, Any], y_col: str, group_cols: List[str]) -> str:
    tc = plan.get("time_column")
    tg = plan.get("time_granularity")
    m = ""
    metrics = plan.get("metrics") or []
    if isinstance(metrics, list) and metrics:
        m0 = metrics[0]
        if isinstance(m0, dict):
            m = f"{m0.get('agg')}({m0.get('column')})→{m0.get('name')}"
    if not m and y_col:
        m = y_col
    g = ",".join(group_cols) if group_cols else "none"
    if tc and tg:
        return f"time={tc}/{tg} | group={g} | metric={m}"
    return f"group={g} | metric={m}"


def _deterministic_finalize_fallback(
    prompt: str,
    fam: str,
    typ: str,
    filters: Any,
    plan: Dict[str, Any],
    x_col: str,
    y_col: str,
    group_cols: List[str],
    unit: Optional[str],
    trend: Dict[str, Any],
    dist: Dict[str, Any],
    verify_sum: Dict[str, Any],
) -> Dict[str, Any]:
    summary_parts: List[str] = []
    filt = _format_filter(filters)

    if fam == "descriptive" and typ == "trend" and isinstance(trend, dict) and trend.get("global"):
        g = trend["global"]
        xb = g.get("x_start"), g.get("x_end")
        npts = g.get("n_points")
        mean = g.get("mean")
        delta = g.get("delta_end_start")
        summary_parts.append(
            f"Computed monthly average {y_col}{(' ('+unit+')') if unit else ''} by {', '.join(group_cols) or 'group'} "
            f"({xb[0]} → {xb[1]}, {npts} points; mean≈{mean:.2f}; Δend-start≈{delta:.2f})."
        )
        insights = []
        insights.append(f"Range: min≈{g.get('min'):.2f}, max≈{g.get('max'):.2f}{(' '+unit) if unit else ''}.")
        if isinstance(trend.get("per_group"), dict) and trend["per_group"]:
            shown = 0
            for k, v in trend["per_group"].items():
                if not isinstance(v, dict):
                    continue
                insights.append(
                    f"{k}: mean≈{v.get('mean'):.2f}, Δend-start≈{v.get('delta_end_start'):.2f}{(' '+unit) if unit else ''}."
                )
                shown += 1
                if shown >= 2:
                    break
        insights = insights[:3]
    elif fam == "descriptive" and typ == "distribution" and isinstance(dist, dict) and dist:
        summary_parts.append(
            f"Computed distribution of {y_col}{(' ('+unit+')') if unit else ''}: "
            f"n={dist.get('n')}, mean≈{dist.get('mean'):.2f}, range≈[{dist.get('min'):.2f}, {dist.get('max'):.2f}]."
        )
        insights = []
    else:
        summary_parts.append(f"Generated analysis for: {prompt[:200]}")
        insights = []

    caveats: List[str] = []
    if isinstance(verify_sum, dict):
        for f in (verify_sum.get("top_findings") or [])[:2]:
            if isinstance(f, dict) and f.get("message"):
                caveats.append(str(f["message"]))
        if verify_sum.get("user_warning"):
            caveats.append(str(verify_sum["user_warning"])[:180])
    caveats = caveats[:2]

    repro = f"filter: {filt} | {_format_agg(plan, y_col, group_cols)}"

    narrative = {
        "summary": " ".join(summary_parts).strip(),
        "insights": insights,
        "caveats": caveats,
        "repro": repro,
    }

    return {
        "status": "ok",
        "confidence": 0.6,
        "narrative": narrative,
        "rationale": "Finalize LLM failed; used deterministic fallback narrative.",
    }


async def step_finalize(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents):
    fam = state.params.get("family", "descriptive")
    typ = state.params.get("type", "distribution")

    viz_spec = state.params.get("viz_spec") or {}
    verify = state.decisions.get("verify") or {}
    verify_sum = _verify_summary(verify)

    aggregate_ctx = state.params.get("aggregate") or {}
    plan = (aggregate_ctx or {}).get("plan") or {}

    x_col = "time_bucket" if "time_bucket" in df.columns else str(plan.get("time_column") or "")
    if not x_col or x_col not in df.columns:
        x_col = "time_bucket" if "time_bucket" in df.columns else ""

    group_cols: List[str] = []
    if isinstance(plan.get("groupby_columns"), list):
        group_cols = [c for c in plan["groupby_columns"] if isinstance(c, str) and c in df.columns]

    y_col = ""
    metrics = plan.get("metrics") or []
    if isinstance(metrics, list) and metrics:
        m0 = metrics[0]
        if isinstance(m0, dict) and m0.get("name") in df.columns:
            y_col = str(m0.get("name"))
    if not y_col:
        for c in df.columns:
            if c == x_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                y_col = c
                break

    unit = None
    units_map = (viz_spec.get("units") or {}) if isinstance(viz_spec, dict) else {}
    if y_col and isinstance(units_map, dict):
        unit = units_map.get(y_col)

    cockpit = _get_aggregation_cockpit(state)
    cockpit_summary = cockpit.get("summary") if isinstance(cockpit, dict) else None

    counts_df = None
    if isinstance(cockpit, dict):
        cdf = cockpit.get("counts_df")
        if isinstance(cdf, pd.DataFrame):
            counts_df = cdf
        if counts_df is None and isinstance(cockpit.get("counts_records"), list):
            try:
                counts_df = pd.DataFrame(cockpit.get("counts_records"))
            except Exception:
                counts_df = None

    counts_examples = {}
    if counts_df is not None and x_col:
        counts_examples = _extract_counts_examples(
            counts_df=counts_df,
            group_cols=group_cols,
            x_col=x_col,
            n_col="n_records",
            top_k=3,
        )

    trend_slim: Dict[str, Any] = {}
    dist_slim: Dict[str, Any] = {}
    trend_ext: Dict[str, Any] = {}
    dist_ext: Dict[str, Any] = {}

    if fam == "descriptive" and typ == "trend":
        trend_slim = _trend_stats(df=df, x_col=x_col, y_col=y_col, group_cols=group_cols, unit=unit, extended=False)
        trend_ext = _trend_stats(df=df, x_col=x_col, y_col=y_col, group_cols=group_cols, unit=unit, extended=True)
    elif fam == "descriptive" and typ == "distribution":
        dist_slim = _distribution_stats(df=df, y_col=y_col, unit=unit, extended=False)
        dist_ext = _distribution_stats(df=df, y_col=y_col, unit=unit, extended=True)

    schema = (df_profile or {}).get("schema") or {}
    data_basis = {
        "final_rows": int(schema.get("rows")) if schema.get("rows") is not None else int(len(df)),
        "final_cols": int(schema.get("cols")) if schema.get("cols") is not None else int(df.shape[1]),
        "n_buckets": int(df[x_col].nunique()) if x_col and x_col in df.columns else None,
        "aggregation_cockpit_summary": cockpit_summary,
        "aggregation_counts_examples": counts_examples,
    }

    stats_for_llm: Dict[str, Any] = {
        "family": fam,
        "type": typ,
        "x_col": x_col,
        "y_col": y_col,
        "group_cols": group_cols,
        "unit": unit,
    }
    if trend_slim:
        stats_for_llm["trend"] = trend_slim
    if dist_slim:
        stats_for_llm["distribution"] = dist_slim

    if fam == "prescriptive" and typ == "decision_formulation":
        ctx = state.params.get("analysis_context") if hasattr(state, "params") else None
        if isinstance(ctx, dict) and ctx:
            stats_for_llm["decision_context"] = ctx

    payload_meta = {
        "family": fam,
        "type": typ,
        "stats": stats_for_llm,
        "data_basis": data_basis,
        "trace": _compact_trace(state),
        "verify_summary": verify_sum,
        "viz_spec": {"title": (viz_spec.get("title") if isinstance(viz_spec, dict) else None)},
        "domain_knowledge_used": _collect_domain_knowledge_used(state),
        "output_constraints": {
            "max_summary_sentences": 2,
            "max_insights": 3,
            "max_caveats": 2,
            "max_next_actions": 3,
            "max_repro_lines": 1,
            "max_chars_per_bullet": 160,
        },
        "preferred_format": {
            "summary": "1-2 sentences",
            "insights": "bullets",
            "caveats": "bullets",
            "repro": "single line",
        },
    }

    state.params["finalize_debug"] = {
        "trend_extended": trend_ext,
        "distribution_extended": dist_ext,
        "verify_full_present": bool(isinstance(verify, dict) and bool(verify)),
    }

    prompt = state.composed_prompt()
    llm_error: Optional[str] = None

    try:
        out = await agents.run(
            "finalize",
            prompt=prompt,
            meta=payload_meta,
            df_profile=df_profile,
        )
        if not isinstance(out, dict):
            raise ValueError("Finalize agent returned non-dict output")
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        filters = state.params.get("filters", [])
        out = _deterministic_finalize_fallback(
            prompt=prompt,
            fam=fam,
            typ=typ,
            filters=filters,
            plan=plan,
            x_col=x_col,
            y_col=y_col,
            group_cols=group_cols,
            unit=unit,
            trend=trend_slim,
            dist=dist_slim,
            verify_sum=verify_sum,
        )
        out["llm_error"] = llm_error

    # ------------------------------------------------------------------
    # Guardrail: if the model returns JSON but the summary is empty/garbled,
    # use deterministic fallback so reporting never prints a blank summary.
    # ------------------------------------------------------------------
    try:
        narr = out.get("narrative") if isinstance(out, dict) else None
        summary_txt = ""
        if isinstance(narr, dict):
            summary_txt = str(narr.get("summary") or "")
        elif isinstance(narr, str):
            summary_txt = narr

        if not str(summary_txt).strip():
            raise ValueError("finalize_empty_summary")

        s = str(summary_txt)
        non_print = sum(1 for ch in s if (ord(ch) < 32 and ch not in "\n\t\r"))
        if len(s) > 20 and (non_print / max(1, len(s))) > 0.02:
            raise ValueError("finalize_garbled_summary")
    except Exception as e_guard:
        if llm_error is None:
            llm_error = f"ValueError: {e_guard}"
        filters = state.params.get("filters", [])
        out = _deterministic_finalize_fallback(
            prompt=prompt,
            fam=fam,
            typ=typ,
            filters=filters,
            plan=plan,
            x_col=x_col,
            y_col=y_col,
            group_cols=group_cols,
            unit=unit,
            trend=trend_slim,
            dist=dist_slim,
            verify_sum=verify_sum,
        )
        out["llm_error"] = llm_error

    state.decisions["finalize"] = out
    state.params["final_narrative"] = out.get("narrative")

    text = ""
    if isinstance(out.get("narrative"), dict):
        text = str(out["narrative"].get("summary") or "")

    meta = {
        "decision": out,
        "rationale": out.get("rationale", ""),
        "text": text,
        "df_delta": None,
        "final_output": out,
    }
    return meta, df