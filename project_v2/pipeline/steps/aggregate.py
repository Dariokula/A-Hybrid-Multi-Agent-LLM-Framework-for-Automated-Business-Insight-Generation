# pipeline/steps/aggregate.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import re

_ALLOWED_GRAN = {"day", "week", "month", "quarter", "year"}
_ALLOWED_AGG = {"count", "sum", "mean", "median", "min", "max"}


# -----------------------------------------------------------------------------
# Time-column preference policy
# -----------------------------------------------------------------------------
def _time_intent_from_prompt(prompt: str) -> str:
    """Return 'start' | 'end' | 'unspecified' based on the prompt."""
    p = (prompt or "").lower()

    start_markers = [
        "start", "begin", "begins", "anfang", "anlauf", "ist_start", "plan_start", "planned start", "actual start",
    ]
    end_markers = [
        "end", "finish", "finished", "complete", "completed", "delivery", "due", "ende", "ist_ende", "plan_ende",
        "planned end", "actual end",
    ]

    has_start = any(m in p for m in start_markers)
    has_end = any(m in p for m in end_markers)
    if has_start and not has_end:
        return "start"
    if has_end and not has_start:
        return "end"
    return "unspecified"


def _score_time_column(col: str, *, intent: str) -> int:
    """Higher score = better default time column."""
    c = (col or "").lower()

    is_actual = any(k in c for k in ["ist_", "actual", "real_"])
    is_planned = any(k in c for k in ["plan", "planned", "soll"])

    is_start = any(k in c for k in ["start", "beginn"])
    is_end = any(k in c for k in ["ende", "end", "finish", "complete"])

    score = 0

    if is_actual:
        score += 100
    elif is_planned:
        score += 60

    if intent == "start":
        score += 25 if is_start else 0
        score += 5 if is_end else 0
    elif intent == "end":
        score += 25 if is_end else 0
        score += 5 if is_start else 0
    else:
        score += 20 if is_end else 0
        score += 10 if is_start else 0

    if any(k in c for k in ["date", "datum", "time", "zeit", "timestamp", "jahr", "monat", "woche", "tag"]):
        score += 3

    return score


def _pick_time_column(state, df: pd.DataFrame, df_profile: Dict[str, Any]) -> Optional[str]:
    prompt = state.composed_prompt() if state is not None else ""
    intent = _time_intent_from_prompt(prompt)

    roles = df_profile.get("roles") or {}
    role_candidates = []
    for t in (roles.get("time_candidates") or []):
        c = (t or {}).get("column")
        if isinstance(c, str) and c in df.columns:
            role_candidates.append(c)

    df_dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    name_hint = [
        c
        for c in df.columns
        if any(h in c.lower() for h in ["jahr", "monat", "date", "time", "start", "ende", "timestamp", "woche", "tag"])
    ]

    pool: List[str] = []
    for c in role_candidates + df_dt + name_hint:
        if c in df.columns and c not in pool:
            pool.append(c)

    if not pool:
        return None

    pool.sort(key=lambda c: _score_time_column(c, intent=intent), reverse=True)
    return pool[0]


def _to_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def _bucket_time(s: pd.Series, gran: str) -> pd.Series:
    if gran == "day":
        return s.dt.floor("D")
    if gran == "week":
        return s.dt.to_period("W").dt.start_time
    if gran == "month":
        return s.dt.to_period("M").dt.start_time
    if gran == "quarter":
        return s.dt.to_period("Q").dt.start_time
    if gran == "year":
        return s.dt.to_period("Y").dt.start_time
    return s


def _infer_granularity_from_prompt(state, time_col: Optional[str]) -> str:
    p = (state.composed_prompt() or "").lower()
    tc = (time_col or "").lower()

    if any(k in tc for k in ["jahr_monat", "year_month", "yyyymm", "month"]):
        return "month"
    if "per month" in p or "monthly" in p or "monat" in p:
        return "month"
    if "per week" in p or "weekly" in p or "woche" in p:
        return "week"
    if "per day" in p or "daily" in p or "tag" in p:
        return "day"
    return "month"


def _explicit_time_preference_in_prompt(prompt: str) -> Dict[str, bool]:
    p = (prompt or "").lower()
    return {
        "explicit_planned": any(k in p for k in ["plan_", "planned", "soll"]),
        "explicit_actual": any(k in p for k in ["ist_", "actual", "real_"]),
        "explicit_start": any(k in p for k in [" start", "begin", "anfang", "beginn", "ist_start", "plan_start"]),
        "explicit_end": any(k in p for k in [" end", "finish", "ende", "ist_ende", "plan_ende", "complete"]),
    }


def _pick_group_columns(state, df: pd.DataFrame) -> List[str]:
    p = (state.composed_prompt() or "").lower()
    group_cols: List[str] = []
    for c in df.columns:
        if c.lower() in p and c not in group_cols:
            group_cols.append(c)
    return group_cols[:3]


def _pick_metric_column(state, df: pd.DataFrame, df_profile: Dict[str, Any]) -> Optional[str]:
    """
    Prefer an already inferred primary metric if present (from columns step),
    otherwise use prompt-name match, then df_profile measure candidates, then first numeric.
    """
    pm = None
    try:
        pm = state.params.get("primary_metric")
    except Exception:
        pm = None
    if isinstance(pm, str) and pm in df.columns and pd.api.types.is_numeric_dtype(df[pm]):
        return pm

    p = (state.composed_prompt() or "").lower()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for c in df.columns:
        if c.lower() in p and c in num_cols:
            return c

    roles = df_profile.get("roles") or {}
    for m in (roles.get("measure_candidates") or []):
        c = (m or {}).get("column")
        if isinstance(c, str) and c in df.columns and c in num_cols:
            return c

    return num_cols[0] if num_cols else None


# -----------------------------------------------------------------------------
# Explicit aggregation request detector (kept strict for diagnostic types)
# -----------------------------------------------------------------------------
def _user_explicitly_requests_aggregation(prompt: str) -> bool:
    p = (prompt or "").lower()

    if any(k in p for k in ["aggregate", "aggregation", "group by", "grouped by", "summarize", "summary by"]):
        return True

    agg_words = r"(average|avg|mean|median|sum|total|count)"
    by_words = r"(by|per|nach|pro|je)"
    if re.search(rf"{agg_words}.*\b{by_words}\b", p):
        return True

    if re.search(r"(im mittel|durchschnitt).*(\bnach\b|\bpro\b|\bje\b)", p):
        return True

    return False


def _fallback_aggregate_plan(df: pd.DataFrame, state, df_profile: Dict[str, Any]) -> Dict[str, Any]:
    a_type = (state.params.get("type") or "").lower()
    fam = (state.params.get("family") or "").lower()
    plan_needed = (a_type == "trend")  # trend requires aggregation by default

    if not plan_needed:
        return {
            "plan_needed": False,
            "plan": {},
            "rationale": "LLM unavailable; no aggregation needed for this analysis type (fallback).",
            "confidence": 0.30,
            "signals": ["llm_failed", "no_aggregation_fallback"],
        }

    time_col = _pick_time_column(state, df, df_profile)
    gran = _infer_granularity_from_prompt(state, time_col)
    group_cols = _pick_group_columns(state, df)
    metric_col = _pick_metric_column(state, df, df_profile)

    metrics = []
    if metric_col:
        metrics.append({"name": f"avg_{metric_col}", "column": metric_col, "agg": "mean"})
    metrics.append({"name": "n_records", "column": None, "agg": "count"})

    return {
        "plan_needed": True,
        "plan": {
            "time_column": time_col,
            "time_granularity": gran,
            "groupby_columns": group_cols,
            "metrics": metrics,
            "sort_by": "time_bucket" if time_col else None,
            "sort_dir": "asc",
            "limit": None,
        },
        "rationale": "LLM unavailable; applied deterministic aggregation plan for trend output (fallback).",
        "confidence": 0.35,
        "signals": ["llm_failed", "fallback_trend_requires_aggregation", f"fallback_gran:{gran}", f"family:{fam}"],
    }


def _build_aggregation_cockpit_counts(
    df: pd.DataFrame,
    *,
    time_col: Optional[str],
    gran: Optional[str],
    group_cols: List[str],
) -> Tuple[Optional[pd.DataFrame], List[str], List[str], Dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, [], [], {"ok": False, "reason": "empty_df"}

    group_cols2 = [c for c in group_cols if c in df.columns]
    group_cols2 = [c for c in group_cols2 if c not in {time_col, "time_bucket"}]

    if time_col and time_col in df.columns and gran in _ALLOWED_GRAN:
        tb = _bucket_time(_to_datetime_series(df, time_col), gran)
        dfc = df.assign(time_bucket=tb)
        keys = ["time_bucket"] + group_cols2
        counts = dfc.groupby(keys, dropna=False).size().rename("n_records").reset_index()
        summary = {
            "ok": True,
            "n_groups": int(len(counts)),
            "total_records": int(counts["n_records"].sum()) if "n_records" in counts.columns else None,
        }
        return counts, ["time_bucket"], group_cols2, summary

    if not group_cols2:
        return None, [], [], {"ok": False, "reason": "no_keys"}

    counts = df.groupby(group_cols2, dropna=False).size().rename("n_records").reset_index()
    summary = {
        "ok": True,
        "n_groups": int(len(counts)),
        "total_records": int(counts["n_records"].sum()) if "n_records" in counts.columns else None,
    }
    return counts, [], group_cols2, summary


def _support_quality_from_counts(counts_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if counts_df is None or not isinstance(counts_df, pd.DataFrame) or counts_df.empty:
        return {"ok": False}
    if "n_records" not in counts_df.columns:
        return {"ok": False}
    s = pd.to_numeric(counts_df["n_records"], errors="coerce").dropna()
    if s.empty:
        return {"ok": False}

    return {
        "ok": True,
        "n_points": int(len(s)),
        "min_n": int(s.min()),
        "p10_n": float(s.quantile(0.10)),
        "median_n": float(s.quantile(0.50)),
        "low_n_lt5": int((s < 5).sum()),
        "low_n_lt10": int((s < 10).sum()),
    }


def _force_trend_aggregation_if_needed(
    *,
    fam: str,
    typ: str,
    plan_needed: bool,
    plan: Dict[str, Any],
    out: Dict[str, Any],
    df: pd.DataFrame,
    state,
    df_profile: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Small models often return plan_needed=False for trend. This is not acceptable because
    the trend analysis contract requires bucketed/aggregated data.
    """
    if fam != "descriptive" or typ != "trend":
        return plan_needed, plan

    # If user explicitly asked "no aggregation" (rare), we could respect it.
    # But for now: trend always aggregates; it's required for an interpretable trend chart.
    if not plan_needed or not isinstance(plan, dict) or not plan:
        time_col = _pick_time_column(state, df, df_profile)
        gran = _infer_granularity_from_prompt(state, time_col)
        group_cols = _pick_group_columns(state, df)
        metric_col = _pick_metric_column(state, df, df_profile)

        metrics: List[Dict[str, Any]] = []
        if metric_col:
            metrics.append({"name": f"avg_{metric_col}", "column": metric_col, "agg": "mean"})
        metrics.append({"name": "n_records", "column": None, "agg": "count"})

        plan2 = {
            "time_column": time_col,
            "time_granularity": gran,
            "groupby_columns": group_cols,
            "metrics": metrics,
            "sort_by": "time_bucket" if time_col else None,
            "sort_dir": "asc",
            "limit": None,
        }

        out.setdefault("auto_adjustments", []).append(
            {
                "reason": "Forced aggregation for descriptive/trend (trend requires time buckets).",
                "forced_plan": {"time_column": time_col, "time_granularity": gran, "groupby_columns": group_cols, "metric": metric_col},
            }
        )
        out["plan_needed"] = True
        out["plan"] = plan2
        return True, plan2

    # Ensure time_column/gran exist for trend
    if not plan.get("time_column"):
        tc = _pick_time_column(state, df, df_profile)
        if tc:
            out.setdefault("auto_adjustments", []).append({"reason": "Trend: filled missing time_column.", "selected": tc})
            plan["time_column"] = tc
    if not plan.get("time_granularity"):
        gran = _infer_granularity_from_prompt(state, plan.get("time_column"))
        out.setdefault("auto_adjustments", []).append({"reason": "Trend: filled missing time_granularity.", "inferred": gran})
        plan["time_granularity"] = gran

    return True, plan


async def step_aggregate(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents):
    rows_before = len(df)
    cols_before = df.shape[1]

    llm_error: Optional[str] = None
    try:
        out = await agents.run(
            "aggregate",
            prompt=state.composed_prompt(),
            meta={
                "step": "aggregate",
                "family": state.params.get("family"),
                "type": state.params.get("type"),
                "filters": state.params.get("filters", []),
                "columns": state.params.get("columns", list(df.columns)),
                "prepare_actions": state.params.get("prepare_actions", []),
            },
            df_profile=df_profile,
        )
        if not isinstance(out, dict):
            raise ValueError("aggregate agent returned non-dict output")
        out.setdefault("llm_error", None)
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        out = _fallback_aggregate_plan(df, state, df_profile)
        out["llm_error"] = llm_error

    fam = (state.params.get("family") or "").lower()
    typ = (state.params.get("type") or "").lower()

    plan_needed = bool(out.get("plan_needed", False))
    plan = out.get("plan") or {}
    if not isinstance(plan, dict):
        plan = {}

    prompt_full = state.composed_prompt() or ""
    explicit_agg = _user_explicitly_requests_aggregation(prompt_full)

    # ---------------------------
    # Hard policies (unchanged)
    # ---------------------------
    if fam == "predictive" and typ in {"regression", "classification"}:
        if plan_needed:
            out.setdefault("auto_adjustments", []).append(
                {"reason": "Predictive ML task: forced plan_needed=False (no aggregation for training)."}
            )
        plan_needed = False
        plan = {}
        out["plan_needed"] = False
        out["plan"] = {}

    if fam == "diagnostic" and typ in {"driver_relationships", "variance_decomposition", "anomaly_explanation"}:
        if not explicit_agg:
            if plan_needed:
                out.setdefault("auto_adjustments", []).append(
                    {
                        "reason": f"{typ}: forced plan_needed=False (keep row-level data). "
                                  "Aggregation is only applied when explicitly requested.",
                        "hint": "Use wording like 'average … by …' or 'group by …' to force aggregation."
                    }
                )
            plan_needed = False
            plan = {}
            out["plan_needed"] = False
            out["plan"] = {}

    if fam == "prescriptive" and typ in {"decision_formulation", "scenario_evaluation", "candidate_ranking"}:
        if plan_needed:
            out.setdefault("auto_adjustments", []).append(
                {"reason": "Prescriptive task: forced plan_needed=False (no aggregation; use row-level data)."}
            )
        plan_needed = False
        plan = {}
        out["plan_needed"] = False
        out["plan"] = {}

    # ---------------------------
    # NEW: enforce aggregation for descriptive trend
    # ---------------------------
    plan_needed, plan = _force_trend_aggregation_if_needed(
        fam=fam, typ=typ, plan_needed=plan_needed, plan=plan, out=out, df=df, state=state, df_profile=df_profile
    )

    df2 = df

    cockpit_counts_df: Optional[pd.DataFrame] = None
    cockpit_key_cols: List[str] = []
    cockpit_group_cols: List[str] = []
    cockpit_summary: Dict[str, Any] = {"ok": False}
    support_quality: Dict[str, Any] = {"ok": False}

    if plan_needed:
        time_col = plan.get("time_column")
        gran = plan.get("time_granularity")
        groupby_cols = plan.get("groupby_columns") or []
        metrics = plan.get("metrics") or []
        sort_by = plan.get("sort_by")
        sort_dir = plan.get("sort_dir", "asc")
        limit = plan.get("limit")

        if time_col and time_col not in df.columns:
            time_col = None

        pref = _explicit_time_preference_in_prompt(prompt_full)
        intent = _time_intent_from_prompt(prompt_full)

        if not time_col:
            picked = _pick_time_column(state, df, df_profile)
            if picked:
                time_col = picked
                plan["time_column"] = picked
                out.setdefault("auto_adjustments", []).append(
                    {"reason": "No time_column provided; selected preferred time column", "selected": picked}
                )

        if time_col:
            tc_l = str(time_col).lower()
            is_planned = any(k in tc_l for k in ["plan", "planned", "soll"])
            if is_planned and not pref.get("explicit_planned", False):
                picked = _pick_time_column(state, df, df_profile)
                if picked and picked != time_col:
                    out.setdefault("auto_adjustments", []).append(
                        {
                            "reason": "Preferred actual timestamp over planned (not explicitly requested)",
                            "before": time_col,
                            "after": picked,
                        }
                    )
                    time_col = picked
                    plan["time_column"] = picked

        if time_col and intent in {"start", "end"}:
            tc_l = str(time_col).lower()
            is_start = "start" in tc_l or "beginn" in tc_l
            is_end = any(k in tc_l for k in ["ende", " end", "finish", "complete"])
            if intent == "start" and not is_start and not pref.get("explicit_end", False):
                picked = _pick_time_column(state, df, df_profile)
                if picked and picked != time_col:
                    out.setdefault("auto_adjustments", []).append(
                        {"reason": "Prompt indicates START; adjusted time_column", "before": time_col, "after": picked}
                    )
                    time_col = picked
                    plan["time_column"] = picked
            if intent == "end" and not is_end and not pref.get("explicit_start", False):
                picked = _pick_time_column(state, df, df_profile)
                if picked and picked != time_col:
                    out.setdefault("auto_adjustments", []).append(
                        {"reason": "Prompt indicates END; adjusted time_column", "before": time_col, "after": picked}
                    )
                    time_col = picked
                    plan["time_column"] = picked

        if time_col and (gran is None or str(gran).lower() == "none"):
            inferred = _infer_granularity_from_prompt(state, time_col)
            gran = inferred
            plan["time_granularity"] = inferred
            out.setdefault("auto_adjustments", []).append(
                {
                    "reason": "LLM returned time_granularity=None; inferred granularity for stable trend contract",
                    "time_column": time_col,
                    "inferred": inferred,
                }
            )

        if gran not in _ALLOWED_GRAN:
            gran = None
            plan["time_granularity"] = None

        groupby_cols = [c for c in groupby_cols if isinstance(c, str) and c in df.columns]
        metrics = [m for m in metrics if isinstance(m, dict)]

        has_count_metric = any(isinstance(m, dict) and str(m.get("agg")).lower() == "count" for m in (metrics or []))
        if not has_count_metric:
            metrics = list(metrics or []) + [{"name": "n_records", "column": None, "agg": "count"}]
            plan["metrics"] = metrics
            out.setdefault("auto_adjustments", []).append(
                {"reason": "Added n_records=count so each aggregate point exposes bucket support."}
            )

        if time_col and gran:
            before = list(groupby_cols)
            groupby_cols = [c for c in groupby_cols if c not in {time_col, "time_bucket"}]
            if before != groupby_cols:
                plan["groupby_columns"] = list(groupby_cols)
                out.setdefault("auto_adjustments", []).append(
                    {
                        "reason": "time_granularity implies bucketing; removed raw time_column from groupby_columns",
                        "time_column": time_col,
                        "before": before,
                        "after": list(groupby_cols),
                    }
                )

        cockpit_counts_df, cockpit_key_cols, cockpit_group_cols, cockpit_summary = _build_aggregation_cockpit_counts(
            df,
            time_col=time_col,
            gran=gran,
            group_cols=groupby_cols,
        )
        support_quality = _support_quality_from_counts(cockpit_counts_df)

        keys: List[str] = []
        df_tmp = df.copy()
        if time_col and gran:
            df_tmp["time_bucket"] = _bucket_time(_to_datetime_series(df_tmp, time_col), gran)
            keys = ["time_bucket"] + groupby_cols
        else:
            keys = groupby_cols

        if keys:
            g = df_tmp.groupby(keys, dropna=False)

            parts: List[pd.Series] = []
            for m in metrics:
                col = m.get("column")
                agg = m.get("agg")
                out_name = m.get("name") or f"{agg}_{col}"

                if agg not in _ALLOWED_AGG:
                    continue

                if agg == "count":
                    ser = g.size().rename(out_name)
                else:
                    if col is None or col not in df_tmp.columns:
                        continue
                    if not pd.api.types.is_numeric_dtype(df_tmp[col]):
                        df_tmp[col] = pd.to_numeric(df_tmp[col], errors="coerce")

                    if agg == "mean":
                        ser = g[col].mean().rename(out_name)
                    elif agg == "sum":
                        ser = g[col].sum().rename(out_name)
                    elif agg == "median":
                        ser = g[col].median().rename(out_name)
                    elif agg == "min":
                        ser = g[col].min().rename(out_name)
                    elif agg == "max":
                        ser = g[col].max().rename(out_name)
                    else:
                        continue

                parts.append(ser)

            if parts:
                df2 = pd.concat(parts, axis=1).reset_index()
            else:
                df2 = df

            ascending = str(sort_dir).lower() != "desc"
            if sort_by and sort_by in df2.columns:
                df2 = df2.sort_values(sort_by, ascending=ascending)
            elif "time_bucket" in df2.columns:
                df2 = df2.sort_values("time_bucket", ascending=True)

            if isinstance(limit, int) and limit > 0:
                df2 = df2.head(limit)

    cockpit = {
        "counts_df": cockpit_counts_df,
        "key_cols": cockpit_key_cols,
        "group_cols": cockpit_group_cols,
        "summary": cockpit_summary,
        "support_quality": support_quality,
    }

    state.params["aggregation_cockpit"] = cockpit
    state.params["aggregate"] = plan if plan_needed else None
    state.decisions["aggregate"] = out

    meta = {
        "decision": out,
        "rationale": out.get("rationale", ""),
        "df_delta": {
            "rows_before": int(rows_before),
            "rows_after": int(len(df2)),
            "cols_before": int(cols_before),
            "cols_after": int(len(df2.columns)),
        },
        "artifacts": {"aggregation_cockpit": cockpit},
    }
    if llm_error:
        meta.setdefault("warnings", []).append(f"aggregate_llm_error: {llm_error}")

    return meta, df2