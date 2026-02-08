# pipeline/steps/prepare.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import pandas as pd
import numpy as np


# -----------------------------
# Helpers: infer keys / intent
# -----------------------------
def _infer_key_candidates(df: pd.DataFrame) -> List[str]:
    keys: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if ("id" in cl) or ("order" in cl) or ("pos" in cl):
            keys.append(c)
    return keys


def _wants_outliers_removed(prompt: str) -> bool:
    p = (prompt or "").lower()
    triggers = [
        "without outliers",
        "remove outliers",
        "no outliers",
        "exclude outliers",
        "ohne ausreißer",
        "ausreißer entfernen",
        "ohne outlier",
        "outlier entfernen",
    ]
    return any(t in p for t in triggers)


def _explicitly_wants_dedup(prompt: str) -> bool:
    p = (prompt or "").lower()
    triggers = ["dedup", "deduplicate", "remove duplicates", "drop duplicates", "duplikat", "duplikate", "duplicate"]
    return any(t in p for t in triggers)


# -----------------------------
# Data quality metrics for prepare
# -----------------------------
def _compute_prepare_quality(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    missing_rate = df.isna().mean().to_dict()

    n_unique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    unique_rate = {c: (n_unique[c] / max(1, n_rows)) for c in df.columns}

    dup_rows = int(df.duplicated().sum())

    key_cols = _infer_key_candidates(df)
    dup_rate_on_keys: Optional[float] = None
    if key_cols:
        dup_rate_on_keys = float(df.duplicated(subset=key_cols).mean())

    dup_rate_all: Optional[float] = None
    if n_cols >= 2:
        dup_rate_all = float(df.duplicated().mean())

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rate = {}
    for c in num_cols[:25]:
        s = df[c].dropna()
        if len(s) < 20:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outlier_rate[c] = float(((s < lo) | (s > hi)).mean())

    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_quality = {}
    for c in cat_cols[:25]:
        s = df[c]
        vc = s.value_counts(dropna=True)
        top_share = float(vc.iloc[0] / max(1, vc.sum())) if len(vc) else 0.0
        cat_quality[c] = {
            "cardinality": int(s.nunique(dropna=True)),
            "top_share": top_share,
            "missing_rate": float(s.isna().mean()),
        }

    return {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "missing_rate": missing_rate,
        "n_unique": n_unique,
        "unique_rate": unique_rate,
        "key_candidates": key_cols,
        "duplicate_rows": dup_rows,
        "duplicate_rate_all": dup_rate_all,
        "duplicate_rate_on_keys": dup_rate_on_keys,
        "numeric_outlier_rate_iqr": outlier_rate,
        "categorical_quality": cat_quality,
    }


def _slim_quality(dq: Dict[str, Any], *, max_cols: int = 10) -> Dict[str, Any]:
    if not isinstance(dq, dict):
        return {}

    missing = dq.get("missing_rate") or {}
    if isinstance(missing, dict):
        missing_items = sorted(missing.items(), key=lambda kv: kv[1], reverse=True)[:max_cols]
        missing_slim = {k: float(v) for k, v in missing_items}
    else:
        missing_slim = {}

    outlier = dq.get("numeric_outlier_rate_iqr") or {}
    if isinstance(outlier, dict):
        outlier_items = sorted(outlier.items(), key=lambda kv: kv[1], reverse=True)[:max_cols]
        outlier_slim = {k: float(v) for k, v in outlier_items}
    else:
        outlier_slim = {}

    catq = dq.get("categorical_quality") or {}
    cat_slim: Dict[str, Any] = {}
    if isinstance(catq, dict):
        items = []
        for c, v in catq.items():
            if isinstance(v, dict):
                items.append((c, int(v.get("cardinality") or 0), v))
        items.sort(key=lambda x: x[1], reverse=True)
        for c, _, v in items[:max_cols]:
            cat_slim[c] = {
                "cardinality": int(v.get("cardinality") or 0),
                "top_share": float(v.get("top_share") or 0.0),
                "missing_rate": float(v.get("missing_rate") or 0.0),
            }

    return {
        "shape": dq.get("shape"),
        "key_candidates": dq.get("key_candidates") or [],
        "duplicate_rows": dq.get("duplicate_rows"),
        "duplicate_rate_all": dq.get("duplicate_rate_all"),
        "duplicate_rate_on_keys": dq.get("duplicate_rate_on_keys"),
        "missing_rate_top": missing_slim,
        "numeric_outlier_rate_iqr_top": outlier_slim,
        "categorical_quality_top": cat_slim,
    }


# -----------------------------
# Deterministic operations
# -----------------------------
def _safe_clip_iqr(df: pd.DataFrame, col: str, k: float) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return df
    lo, hi = q1 - k * iqr, q3 + k * iqr
    df2 = df.copy()
    df2[col] = s.clip(lower=lo, upper=hi)
    return df2


def _is_metric_only_dataframe(df: pd.DataFrame) -> bool:
    if df.shape[1] != 1:
        return False
    col = df.columns[0]
    return pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])


_DISGUISED_MISSING = {"", " ", "na", "n/a", "null", "none", "nan", "-", "--", "?"}


def _normalize_disguised_missing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Replace common disguised-missing tokens with NA in text-like columns.
    This is a standard enterprise cleaning step and improves later grouping/aggregation stability.
    """
    if not cols:
        return df
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            continue
        s = df2[c]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s)):
            continue
        s2 = s.astype("string")
        norm = s2.str.strip().str.lower()
        df2[c] = s2.mask(norm.isin(_DISGUISED_MISSING))
    return df2


def _fallback_prepare_decision(dq: Dict[str, Any], prompt: str, df_profile: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    """
    Conservative deterministic fallback (analysis-ready contract focus):
    - normalize disguised missing tokens (text columns)
    - strip strings (reduces encoding fragmentation without semantic loss)
    """
    actions: List[Dict[str, Any]] = []

    actions.append({"name": "normalize_missing_tokens", "params": {}})
    actions.append({"name": "strip_strings", "params": {}})

    return {
        "actions": actions,
        "rationale": "LLM prepare call failed; applying safe, non-destructive cleaning (normalize missing tokens + strip strings).",
        "confidence": 0.35,
        "signals": ["llm_failed", "safe_fallback_cleaning"],
    }


async def _safe_run_prepare_agent(
    *,
    agents,
    prompt_txt: str,
    meta: Dict[str, Any],
    df_profile: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        out = await agents.run("prepare", prompt=prompt_txt, meta=meta, df_profile=df_profile)
        if not isinstance(out, dict):
            raise ValueError("prepare agent returned non-dict output")
        out["llm_error"] = None
        return out
    except Exception as e:
        return {
            "actions": [],
            "rationale": "LLM prepare call failed; proceeding with deterministic fallback.",
            "confidence": 0.25,
            "signals": ["llm_failed"],
            "llm_error": f"{type(e).__name__}: {e}",
        }


# -----------------------------
# Step
# -----------------------------
async def step_prepare(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents):
    rows_before, cols_before = df.shape

    prompt_txt = state.composed_prompt()
    wants_outliers = _wants_outliers_removed(prompt_txt)
    wants_dedup = _explicitly_wants_dedup(prompt_txt)

    dq_full = _compute_prepare_quality(df)
    dq_slim = _slim_quality(dq_full, max_cols=10)

    allowed_actions_default: Set[str] = {
        "drop_duplicates",
        "strip_strings",
        "coerce_numeric",
        "parse_datetimes",
        "impute_numeric_median",
        "impute_categorical_mode",
        "drop_rows_missing_required",
        "clip_outliers_iqr",
        # new safe action (deterministic only)
        "normalize_missing_tokens",
    }

    max_drop_frac = 0.10
    outlier_iqr_k_default = 1.5

    # Keep LLM payload small; rely on df_profile for semantics if needed.
    meta_for_llm = {
        "step": "prepare",
        "family": state.params.get("family"),
        "type": state.params.get("type"),
        "filters": state.params.get("filters", []),
        "columns": state.params.get("columns", list(df.columns)),
        "data_quality": dq_slim,
        "allowed_actions": sorted(list(allowed_actions_default)),
        "notes": {
            "metric_only_dataframe": _is_metric_only_dataframe(df),
            "wants_outliers_removed": wants_outliers,
            "wants_dedup": wants_dedup,
            "max_row_drop_fraction": max_drop_frac,
            "outlier_iqr_k_default": outlier_iqr_k_default,
            # prep contract hint (thesis-relevant)
            "goal": "analysis_ready_contract: stable types/encodings, non-misleading aggregates",
        },
        "output_constraints": {
            "max_actions": 5,
            "max_rationale_chars": 260,
        },
    }

    out = await _safe_run_prepare_agent(
        agents=agents,
        prompt_txt=prompt_txt,
        meta=meta_for_llm,
        df_profile=df_profile,
    )

    selected_cols = state.params.get("columns", list(df.columns))

    # If LLM failed, do deterministic minimal cleaning.
    if out.get("llm_error"):
        out = {
            **_fallback_prepare_decision(dq_full, prompt_txt, df_profile, selected_cols),
            "llm_error": out.get("llm_error"),
        }

    actions = out.get("actions", []) or []
    df2 = df

    # Deterministic pre-cleaning that is almost always beneficial and cheap:
    # If encoding fragmentation was detected in profiling, stripping strings is safe.
    # Also normalize disguised missing tokens early for text-like columns.
    text_cols = df2.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if text_cols:
        df2 = _normalize_disguised_missing(df2, text_cols)

    # Apply deterministic actions with guardrails
    for a in actions:
        if not isinstance(a, dict):
            continue

        name = (a.get("name") or "").strip()
        params = a.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if name not in allowed_actions_default:
            continue

        # --- normalize_missing_tokens ---
        if name == "normalize_missing_tokens":
            obj_cols = df2.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            if obj_cols:
                df2 = _normalize_disguised_missing(df2, obj_cols)

        # --- drop_duplicates ---
        elif name == "drop_duplicates":
            if _is_metric_only_dataframe(df2) and not wants_dedup:
                continue

            key_cols = dq_full.get("key_candidates") or []
            subset = params.get("subset") or key_cols
            if not isinstance(subset, list):
                subset = key_cols
            subset = [c for c in subset if c in df2.columns]

            if not subset:
                if wants_dedup and df2.shape[1] >= 2:
                    subset = list(df2.columns)
                else:
                    continue

            before = len(df2)
            df_tmp = df2.drop_duplicates(subset=subset)
            drop_frac = (before - len(df_tmp)) / max(1, before)

            if drop_frac <= max_drop_frac or wants_dedup:
                df2 = df_tmp

        # --- strip_strings ---
        elif name == "strip_strings":
            obj_cols = df2.select_dtypes(include=["object", "string"]).columns
            if len(obj_cols) == 0:
                continue
            df2 = df2.copy()
            for c in obj_cols:
                df2[c] = df2[c].where(df2[c].isna(), df2[c].astype(str).str.strip())

        # --- coerce_numeric ---
        elif name == "coerce_numeric":
            cols = params.get("columns") or []
            if not isinstance(cols, list):
                cols = []
            cols = [c for c in cols if c in df2.columns]
            if not cols:
                continue
            df2 = df2.copy()
            for c in cols:
                df2[c] = pd.to_numeric(df2[c], errors="coerce")

        # --- parse_datetimes ---
        elif name == "parse_datetimes":
            cols = params.get("columns") or []
            if not isinstance(cols, list):
                cols = []
            cols = [c for c in cols if c in df2.columns]
            if not cols:
                continue
            df2 = df2.copy()
            for c in cols:
                df2[c] = pd.to_datetime(df2[c], errors="coerce")

        # --- impute_numeric_median ---
        elif name == "impute_numeric_median":
            cols = params.get("columns") or []
            if not isinstance(cols, list):
                cols = []
            cols = [c for c in cols if c in df2.columns]
            if not cols:
                continue
            df2 = df2.copy()
            for c in cols:
                s = pd.to_numeric(df2[c], errors="coerce")
                if s.isna().mean() == 0:
                    continue
                df2[c] = s.fillna(s.median())

        # --- impute_categorical_mode ---
        elif name == "impute_categorical_mode":
            cols = params.get("columns") or []
            if not isinstance(cols, list):
                cols = []
            cols = [c for c in cols if c in df2.columns]
            if not cols:
                continue
            df2 = df2.copy()
            for c in cols:
                if df2[c].isna().mean() == 0:
                    continue
                vc = df2[c].value_counts(dropna=True)
                if len(vc):
                    df2[c] = df2[c].fillna(vc.index[0])

        # --- drop_rows_missing_required ---
        elif name == "drop_rows_missing_required":
            cols = params.get("columns") or []
            if not isinstance(cols, list):
                cols = []
            cols = [c for c in cols if c in df2.columns]
            if not cols:
                continue
            before = len(df2)
            df_tmp = df2.dropna(subset=cols)
            drop_frac = (before - len(df_tmp)) / max(1, before)
            if drop_frac <= max_drop_frac:
                df2 = df_tmp

        # --- clip_outliers_iqr ---
        elif name == "clip_outliers_iqr":
            if not wants_outliers:
                continue
            col = params.get("column")
            if not col or col not in df2.columns:
                continue
            k = params.get("k", outlier_iqr_k_default)
            try:
                k = float(k)
            except Exception:
                k = outlier_iqr_k_default
            df2 = _safe_clip_iqr(df2, col, k)

    rows_after, cols_after = df2.shape

    state.params["prepare_actions"] = actions
    state.decisions["prepare"] = {
        **out,
        "data_quality": dq_full,
        "data_quality_slim_sent": dq_slim,
        # contract-ish traceability
        "contract": {
            "columns": list(df2.columns),
            "dtypes": {c: str(df2[c].dtype) for c in df2.columns},
            "rows": int(rows_after),
        },
    }

    meta = {
        "decision": state.decisions["prepare"],
        "rationale": (out.get("rationale", "") or ""),
        "df_delta": {
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "cols_before": int(cols_before),
            "cols_after": int(cols_after),
        },
    }
    return meta, df2