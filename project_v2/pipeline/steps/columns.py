# pipeline/steps/columns.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Set, Tuple
import re
import pandas as pd

from config import MAX_COLUMNS_DEFAULT

_DATETIME_HINTS = ("date", "datum", "time", "zeit", "start", "ende", "timestamp", "jahr", "monat", "woche", "tag")
_TREND_HINTS = ("trend", "weekly", "month", "monthly", "daily", "year", "jah", "jahr", "woche", "monat", "tag", "per month", "per week")
_GROUP_HINTS = ("by", "per", "nach", "group", "grouped", "breakdown", "split", "segment", "status", "type", "klasse", "gruppe")

_ID_NAME_HINTS = ("id", "order", "auftrag", "pos", "position", "nr", "nummer", "key")


# -----------------------------------------------------------------------------
# Time-column preference policy
# -----------------------------------------------------------------------------
def _time_intent_from_prompt(prompt: str) -> str:
    p = (prompt or "").lower()
    start_markers = [
        "start", "begin", "begins", "anfang",
        "ist_start", "plan_start", "planned start", "actual start",
    ]
    end_markers = [
        "end", "finish", "finished", "complete", "completed",
        "delivery", "due", "ende",
        "ist_ende", "plan_ende", "planned end", "actual end",
    ]
    has_start = any(m in p for m in start_markers)
    has_end = any(m in p for m in end_markers)
    if has_start and not has_end:
        return "start"
    if has_end and not has_start:
        return "end"
    return "unspecified"


def _score_time_column(col: str, *, intent: str) -> int:
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
        # default to END if unspecified
        score += 20 if is_end else 0
        score += 10 if is_start else 0

    if any(k in c for k in ["date", "datum", "time", "zeit", "timestamp", "jahr", "monat", "woche", "tag"]):
        score += 3

    return score


def _infer_datetime_candidates(df_profile: Dict[str, Any], *, prompt: str = "") -> List[str]:
    schema = (df_profile.get("schema") or {})
    dtypes = (schema.get("dtypes") or {})

    dt = [c for c, t in dtypes.items() if "datetime" in str(t)]
    if not dt:
        dt = [c for c in dtypes.keys() if any(h in c.lower() for h in _DATETIME_HINTS)]
    if not dt:
        return []

    intent = _time_intent_from_prompt(prompt)
    return sorted(dt, key=lambda c: _score_time_column(c, intent=intent), reverse=True)


def _filter_columns_exist(cols: List[str], df_cols: List[str]) -> List[str]:
    s = set(df_cols)
    out: List[str] = []
    for c in cols:
        if c in s and c not in out:
            out.append(c)
    return out


def _tokenize_prompt(prompt: str) -> Set[str]:
    p = (prompt or "").lower()
    return set(re.findall(r"[a-zA-Z_]+", p))


def _uninformative_from_profile(df_profile: Dict[str, Any]) -> Set[str]:
    u = df_profile.get("uninformative") or {}
    cols = set()
    for item in (u.get("mostly_empty") or []):
        if isinstance(item, dict) and item.get("column"):
            cols.add(str(item["column"]))
    for item in (u.get("near_constant") or []):
        if isinstance(item, dict) and item.get("column"):
            cols.add(str(item["column"]))
    return cols


def _flatten_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            parts.append(str(k))
            parts.append(_flatten_text(v))
        return "\n".join([p for p in parts if p])
    if isinstance(x, (list, tuple)):
        return "\n".join([_flatten_text(v) for v in x])
    return str(x)


def _extract_field_descriptions_from_text(text: str) -> Dict[str, str]:
    """Extract '<column>: <description>' or '<column> - <description>' pairs from DK text."""
    t = (text or "").strip()
    if not t:
        return {}

    out: Dict[str, str] = {}

    for m in re.finditer(r"(?m)^\s*[•\-\*]?\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$", t):
        col = m.group(1).strip()
        desc = m.group(2).strip()
        if col and desc and col not in out:
            out[col] = desc

    for m in re.finditer(r"(?m)^\s*[•\-\*]?\s*([A-Za-z0-9_]+)\s*-\s*(.+?)\s*$", t):
        col = m.group(1).strip()
        desc = m.group(2).strip()
        if col and desc and col not in out:
            out[col] = desc

    return out


def _tokenize_identifier(name: str) -> Set[str]:
    s = (name or "").strip().lower()
    if not s:
        return set()
    parts = re.split(r"[_\-\s]+", s)
    toks: Set[str] = set()
    for p in parts:
        if not p:
            continue
        toks.add(p)
        toks |= set(re.findall(r"[a-z]+|\d+", p))
    return {t for t in toks if t}


def _metric_intent_tokens(prompt: str) -> Set[str]:
    p = (prompt or "").lower()
    tokens: Set[str] = set()

    has_dev = any(k in p for k in ["deviation", "abweich", "delta", "variance", "gap", "miss", "late", "delay", "terminabweich"])
    if has_dev:
        tokens |= {"deviation", "abweich", "delta", "delay", "late", "gap", "terminabweich"}

    has_sched = any(k in p for k in ["schedule", "planned", "plan", "soll", "termin"])
    if has_sched:
        tokens |= {"schedule", "planned", "plan", "termin", "soll"}

    has_start = any(k in p for k in ["start", "begin", "anfang", "zugang"])

    # Distinguish end-as-time (metric intent) vs finished-as-status (filter intent)
    has_end_time = any(k in p for k in [
        "end", "ende", "abgang", "due", "delivery",
        "ist_ende", "plan_ende", "planned end", "actual end",
    ])
    has_end_status = any(k in p for k in [
        "finish", "finished", "complete", "completed",
        "fertig", "fertiggemeldet"
    ])

    # Only treat status words as end-intent if prompt is clearly schedule/deviation-oriented
    has_end = has_end_time or (has_end_status and (has_dev or has_sched))

    if has_start and not has_end:
        tokens |= {"start", "begin", "anfang", "zugang"}
    elif has_end and not has_start:
        tokens |= {"end", "finish", "ende", "abgang"}
    else:
        # unspecified: do NOT inject start/end intent
        pass

    # Lead/cycle time tokens: keep, but avoid letting "dlz" overpower deviation prompts
    if any(k in p for k in ["lead time", "durchlauf", "run-through", "run through", "dlz"]):
        tokens |= {"lead", "leadtime", "durchlauf", "run"}
        if ("dlz" in p) or (not has_dev):
            tokens.add("dlz")

    if any(k in p for k in ["cycle time", "cycle", "dlz"]):
        tokens |= {"cycle", "cycletime"}
        if ("dlz" in p) or (not has_dev):
            tokens.add("dlz")

    return tokens


def _score_metric_candidate(*, col: str, prompt_tokens: Set[str], dk_field_desc: Dict[str, str]) -> float:
    col_toks = _tokenize_identifier(col)
    score = 0.0

    score += 2.5 * len(col_toks & prompt_tokens)

    desc = dk_field_desc.get(col) or dk_field_desc.get(col.strip()) or ""
    if desc:
        desc_toks = _tokenize_identifier(desc)
        score += 1.4 * len(desc_toks & prompt_tokens)

        if ("end" in prompt_tokens or "finish" in prompt_tokens or "ende" in prompt_tokens or "abgang" in prompt_tokens) and any(
            k in desc.lower() for k in ["end", "finish", "ende", "abgang"]
        ):
            score += 1.0
        if ("start" in prompt_tokens or "begin" in prompt_tokens or "anfang" in prompt_tokens or "zugang" in prompt_tokens) and any(
            k in desc.lower() for k in ["start", "begin", "anfang", "zugang"]
        ):
            score += 1.0

        if ("schedule" in prompt_tokens or "termin" in prompt_tokens or "plan" in prompt_tokens) and any(
            k in desc.lower() for k in ["schedule", "planned", "plan", "termin", "soll"]
        ):
            score += 0.6

    if ("deviation" in prompt_tokens or "abweich" in prompt_tokens or "delta" in prompt_tokens or "terminabweich" in prompt_tokens):
        if any(k in col_toks for k in ["ta", "dev", "deviation", "abweich", "delta"]):
            score += 0.6

    score += max(0.0, 0.2 - (len(str(col)) / 120.0))
    return score


def _get_domain_knowledge_text_for_columns(state) -> str:
    """Prefer per-step DK stored in state (engine), else latest DK."""
    dk = None
    dk_steps = state.params.get("_domain_knowledge_steps")
    if isinstance(dk_steps, dict):
        dk = dk_steps.get("columns")
    if dk is None:
        dk = state.params.get("_domain_knowledge_latest")
    return _flatten_text(dk)


def _infer_primary_metric(*, df: pd.DataFrame, state) -> Tuple[Optional[str], float, Dict[str, float]]:
    """
    Infer the best-matching numeric metric column for the prompt.

    Returns: (best_col, best_score, score_map)
    """
    prompt = state.composed_prompt() or ""
    prompt_tokens = _metric_intent_tokens(prompt)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return None, -1.0, {}

    dk_text = _get_domain_knowledge_text_for_columns(state)
    dk_field_desc = _extract_field_descriptions_from_text(dk_text)

    score_map: Dict[str, float] = {}
    best_col = None
    best_score = -1.0

    for c in num_cols:
        sc = _score_metric_candidate(col=str(c), prompt_tokens=prompt_tokens, dk_field_desc=dk_field_desc)
        score_map[str(c)] = float(sc)
        if sc > best_score:
            best_score = sc
            best_col = str(c)

    return best_col, float(best_score), score_map


def _should_infer_primary_metric(state) -> bool:
    fam = (state.params.get("family") or "").strip().lower()
    typ = (state.params.get("type") or "").strip().lower()
    if fam == "prescriptive" and typ == "decision_formulation":
        return False
    return True


def _reorder_columns_with_primary_metric_first(df: pd.DataFrame, cols: List[str], primary: Optional[str]) -> List[str]:
    if not primary or primary not in cols:
        return cols

    numeric = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    nonnum = [c for c in cols if c not in set(numeric)]

    if primary not in numeric:
        return cols

    numeric2 = [primary] + [c for c in numeric if c != primary]
    return nonnum + numeric2 if nonnum else numeric2


def _is_id_like(col: str) -> bool:
    cl = (col or "").lower()
    return any(h in cl for h in _ID_NAME_HINTS)


def _is_time_like(col: str) -> bool:
    cl = (col or "").lower()
    return any(h in cl for h in _DATETIME_HINTS) or any(x in cl for x in ("start", "ende", "end"))


def _ensure_primary_metric_in_columns(*, df: pd.DataFrame, cols: List[str], primary: str, max_cols: int) -> List[str]:
    """Ensure primary metric is present; if we need to make space, drop low-value columns first."""
    if primary in cols:
        return cols

    cols2 = list(cols)
    if len(cols2) < max_cols:
        cols2.append(primary)
        return cols2

    drop_idx = None
    for i, c in enumerate(cols2):
        if _is_id_like(c):
            drop_idx = i
            break
    if drop_idx is None:
        for i, c in enumerate(cols2):
            if _is_time_like(c):
                drop_idx = i
                break
    if drop_idx is None:
        drop_idx = len(cols2) - 1

    cols2[drop_idx] = primary
    return cols2


# -----------------------------------------------------------------------------
# Contract enforcement for small / unreliable LLM outputs
# -----------------------------------------------------------------------------
def _analysis_needs_time_column(family: str, typ: str) -> bool:
    f = (family or "").lower().strip()
    t = (typ or "").lower().strip()
    return (t in {"trend", "forecasting"}) or (f == "predictive" and t == "forecasting")


def _analysis_needs_many_features(family: str, typ: str) -> bool:
    f = (family or "").lower().strip()
    t = (typ or "").lower().strip()
    if f == "diagnostic" and t in {"driver_relationships", "variance_decomposition", "anomaly_explanation"}:
        return True
    if f == "predictive" and t in {"regression", "classification"}:
        return True
    return False


def _explicitly_mentions_column(prompt_l: str, col: str) -> bool:
    """Loose mention check: also matches tokenized variants like 'resource group' for 'resource_group'."""
    if not col:
        return False
    c = col.lower()
    if c in prompt_l:
        return True
    parts = [p for p in re.split(r"[_\-]+", c) if p]
    if len(parts) >= 2:
        if " ".join(parts) in prompt_l:
            return True
    return False


def _rank_numeric_candidates(
    *,
    df_profile: Dict[str, Any],
    numeric_cols: List[str],
    primary: Optional[str],
    score_map: Dict[str, float],
) -> List[str]:
    """Rank numeric columns by (primary first) then score_map then basic profile quality."""
    if not numeric_cols:
        return []

    missing_rate = (df_profile.get("missing_rate") or {}) if isinstance(df_profile, dict) else {}
    numeric_prof = (df_profile.get("numeric") or {}) if isinstance(df_profile, dict) else {}

    def key(c: str):
        mr = float(missing_rate.get(c, 0.0) or 0.0)
        sc = float(score_map.get(c, 0.0) or 0.0)
        std = 0.0
        try:
            if isinstance(numeric_prof.get(c), dict) and numeric_prof[c].get("std") is not None:
                std = float(numeric_prof[c].get("std") or 0.0)
        except Exception:
            std = 0.0
        is_primary = 1 if (primary and c == primary) else 0
        return (-is_primary, -sc, mr, -std, c)

    return sorted(list(dict.fromkeys(numeric_cols)), key=key)


def _enforce_column_contract(
    *,
    df: pd.DataFrame,
    df_profile: Dict[str, Any],
    state,
    cols: List[str],
    primary_metric: Optional[str],
    score_map: Dict[str, float],
) -> List[str]:
    """Trim obviously unhelpful columns for the chosen analysis type."""
    fam = (state.params.get("family") or "").lower().strip() if hasattr(state, "params") else ""
    typ = (state.params.get("type") or "").lower().strip() if hasattr(state, "params") else ""
    prompt_l = (state.composed_prompt() or "").lower()

    cols0 = [c for c in cols if isinstance(c, str) and c in df.columns]
    if not cols0:
        return cols

    if _analysis_needs_many_features(fam, typ):
        out = []
        for c in cols0:
            if _is_id_like(c) and not _explicitly_mentions_column(prompt_l, c):
                continue
            out.append(c)
        return out or cols0

    needs_time = _analysis_needs_time_column(fam, typ)

    if fam == "descriptive" and typ in {"stats_summary", "distribution"}:
        numeric_cols = [c for c in cols0 if pd.api.types.is_numeric_dtype(df[c])]

        ranked = _rank_numeric_candidates(
            df_profile=df_profile,
            numeric_cols=numeric_cols,
            primary=primary_metric,
            score_map=score_map,
        )

        keep_n = 1 if (primary_metric and primary_metric in ranked) else 3
        kept_numeric = ranked[:keep_n] if ranked else []

        group_cols: List[str] = []
        if any(k in prompt_l for k in _GROUP_HINTS):
            for c in cols0:
                if c in kept_numeric:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    continue
                if _is_id_like(c) or _is_time_like(c):
                    continue
                if _explicitly_mentions_column(prompt_l, c):
                    group_cols.append(c)
                    break

        out = kept_numeric + group_cols
        return out or (kept_numeric or ranked or cols0)

    if fam == "descriptive" and typ in {"group_compare", "relationships"}:
        out = []
        for c in cols0:
            if _is_id_like(c) and not _explicitly_mentions_column(prompt_l, c):
                continue
            if not needs_time and _is_time_like(c) and not _explicitly_mentions_column(prompt_l, c):
                continue
            out.append(c)
        return out or cols0

    out = []
    for c in cols0:
        if _is_id_like(c) and not _explicitly_mentions_column(prompt_l, c):
            continue
        if not needs_time and _is_time_like(c) and not _explicitly_mentions_column(prompt_l, c):
            continue
        out.append(c)
    return out or cols0


def _deterministic_fallback_columns(df: pd.DataFrame, df_profile: Dict[str, Any], state) -> Dict[str, Any]:
    df_cols = list(df.columns)
    prompt = state.composed_prompt() or ""
    p_l = prompt.lower()
    toks = _tokenize_prompt(prompt)

    kept: List[str] = []

    if any(k in p_l for k in _TREND_HINTS):
        for c in _infer_datetime_candidates(df_profile, prompt=prompt):
            if c in df_cols:
                kept.append(c)
                break

    if any(k in p_l for k in _GROUP_HINTS):
        for c in df_cols:
            cl = c.lower()
            if cl in toks or cl in p_l:
                if c not in kept:
                    kept.append(c)

    primary, sc, _ = _infer_primary_metric(df=df, state=state)
    if primary is not None and primary in df_cols and sc >= 1.6 and primary not in kept:
        kept.append(primary)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in num_cols:
        if c not in kept:
            kept.append(c)

    if not kept:
        kept = df_cols[: min(6, len(df_cols))]

    max_cols = int(state.params.get("max_columns", MAX_COLUMNS_DEFAULT) or MAX_COLUMNS_DEFAULT)
    final_cols = kept[:max_cols]

    return {
        "columns": final_cols,
        "rationale": "LLM columns selection failed; used conservative deterministic fallback (time/group/primary metric/numeric).",
        "confidence": 0.35,
        "signals": ["llm_failed", "fallback_time_group_metric", "primary_metric_inference"],
    }


async def _safe_run_columns_agent(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents) -> Dict[str, Any]:
    llm_error: Optional[str] = None
    try:
        out = await agents.run(
            "columns",
            prompt=state.composed_prompt(),
            meta={
                "step": "columns",
                "family": state.params.get("family"),
                "type": state.params.get("type"),
                "filters": state.params.get("filters", []),
                "output_constraints": {
                    "max_columns": int(state.params.get("max_columns", MAX_COLUMNS_DEFAULT) or MAX_COLUMNS_DEFAULT),
                    "max_rationale_chars": 280,
                },
            },
            df_profile=df_profile,
        )
        if not isinstance(out, dict):
            raise ValueError("columns agent returned non-dict output")
        out["llm_error"] = None
        return out
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        out = _deterministic_fallback_columns(df=df, df_profile=df_profile, state=state)
        out["llm_error"] = llm_error
        return out


async def step_columns(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents):
    out = await _safe_run_columns_agent(df=df, state=state, df_profile=df_profile, agents=agents)

    df_cols = list(df.columns)
    prompt = state.composed_prompt() or ""
    prompt_l = prompt.lower()

    kept: List[str] = []

    if any(k in prompt_l for k in _TREND_HINTS):
        for c in _infer_datetime_candidates(df_profile, prompt=prompt):
            if c in df_cols and c not in kept:
                kept.append(c)
                break

    for c in df_cols:
        if c.lower() in prompt_l and c not in kept:
            kept.append(c)

    chosen = _filter_columns_exist(out.get("columns", []) or [], df_cols)

    final_cols: List[str] = []
    for c in kept + chosen:
        if c in df_cols and c not in final_cols:
            final_cols.append(c)

    if not final_cols:
        final_cols = df_cols[: min(6, len(df_cols))]

    uninf = _uninformative_from_profile(df_profile)
    final_cols = [c for c in final_cols if c not in uninf or (c.lower() in prompt_l)]

    max_cols = int(state.params.get("max_columns", MAX_COLUMNS_DEFAULT) or MAX_COLUMNS_DEFAULT)
    if len(final_cols) > max_cols:
        kept_set = set(kept)
        trimmed = [c for c in final_cols if c in kept_set]
        rest = [c for c in final_cols if c not in kept_set]
        final_cols = trimmed + rest[: max(0, max_cols - len(trimmed))]

    # -------------------------------
    # Domain-knowledge-aware primary-metric guard
    # -------------------------------
    primary_metric = None
    primary_score = -1.0
    score_map: Dict[str, float] = {}

    if _should_infer_primary_metric(state):
        primary_metric, primary_score, score_map = _infer_primary_metric(df=df, state=state)

        if primary_metric is not None and primary_metric in df_cols and primary_score >= 1.6:
            state.params["primary_metric"] = primary_metric
            state.params["primary_metric_score"] = float(primary_score)

            final_cols = _ensure_primary_metric_in_columns(
                df=df,
                cols=final_cols,
                primary=primary_metric,
                max_cols=max_cols,
            )

            final_cols = _reorder_columns_with_primary_metric_first(df, final_cols, primary_metric)
        else:
            state.params.pop("primary_metric", None)
            state.params.pop("primary_metric_score", None)
            primary_metric = None

    # -------------------------------
    # Enforce a minimal, type-aware column contract.
    # -------------------------------
    final_cols = _enforce_column_contract(
        df=df,
        df_profile=df_profile,
        state=state,
        cols=final_cols,
        primary_metric=primary_metric,
        score_map=score_map,
    )

    cols_before = len(df_cols)
    cols_after = len(final_cols)
    dropped = [c for c in df_cols if c not in set(final_cols)]

    df2 = df[final_cols].copy()

    state.params["columns"] = final_cols
    state.decisions["columns"] = {
        **out,
        "columns": final_cols,
        "kept_columns": kept,
        "max_columns": max_cols,
        "uninformative_dropped": sorted(list(set(df_cols) & _uninformative_from_profile(df_profile) - set(final_cols))),
        "primary_metric": primary_metric,
        "primary_metric_score": (None if primary_metric is None else float(primary_score)),
        "primary_metric_score_map_top": dict(sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:8]),
        "signals": list(dict.fromkeys(list(out.get("signals") or []) + (["primary_metric_guard"] if primary_metric else []))),
    }

    meta = {
        "decision": state.decisions["columns"],
        "rationale": (state.decisions["columns"].get("rationale") or ""),
        "df_delta": {"cols_before": cols_before, "cols_after": cols_after, "dropped": dropped},
    }
    return meta, df2