# pipeline/profiling.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import re
import warnings

import pandas as pd
import numpy as np

from config import (
    PROFILE_MAX_CATEGORICAL_COLS,
    PROFILE_TOP_K,
    PROFILE_SAMPLE_ROWS,
)

# -----------------------------
# Helpers
# -----------------------------

_ID_NAME_HINTS = ("id", "order", "auftrag", "pos", "position", "nr", "nummer", "key")
_TIME_NAME_HINTS = ("date", "datum", "time", "zeit", "start", "ende", "timestamp", "jahr", "monat", "woche", "tag")
_BOOL_LEVELS = {"0", "1", "y", "n", "yes", "no", "true", "false", "t", "f"}


def _safe_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=0)


def _parse_datetime_best_effort(s: pd.Series) -> pd.Series:
    """
    Robust datetime parsing for profiling:
    - Avoid deprecated infer_datetime_format.
    - Try fast/strict-ish parsers first (ISO8601 / mixed) if supported by pandas version.
    - Suppress noisy warnings that are expected during heuristic detection.
    """
    # Work on strings, small sample expected
    s0 = s.astype("string")

    with warnings.catch_warnings():
        # These warnings are not actionable in a heuristic "candidate check".
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # 1) Try ISO8601 (fast & consistent) if supported
        try:
            dt = pd.to_datetime(s0, errors="coerce", utc=False, format="ISO8601")
            return dt
        except TypeError:
            pass
        except Exception:
            pass

        # 2) Try mixed (handles heterogeneous formats) if supported
        try:
            dt = pd.to_datetime(s0, errors="coerce", utc=False, format="mixed")
            return dt
        except TypeError:
            pass
        except Exception:
            pass

        # 3) Fallback: let pandas/dateutil decide (may be slower, but warnings suppressed)
        dt = pd.to_datetime(s0, errors="coerce", utc=False)
        return dt


def _is_likely_datetime_series(s: pd.Series) -> Tuple[bool, float]:
    """
    Try to parse a sample of a non-datetime series to datetime.
    Returns (is_candidate, parse_rate).
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return True, 1.0

    # only attempt on object/string-like
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False, 0.0

    s0 = s.dropna()
    if s0.empty:
        return False, 0.0

    # cap rows for speed
    s0 = s0.head(200)
    parsed = _parse_datetime_best_effort(s0)
    rate = float(parsed.notna().mean())
    return (rate >= 0.9), rate


def _top_value_counts(s: pd.Series, k: int) -> List[Dict[str, Any]]:
    vc = s.astype("object").value_counts(dropna=False).head(k)
    out = []
    for key, val in vc.items():
        out.append({"value": (None if pd.isna(key) else str(key)), "count": int(val)})
    return out


def _encoding_consistency_flags(s: pd.Series) -> Dict[str, Any]:
    """
    Detect fragmentation due to case/whitespace and simple boolean-code mixtures.
    Keep this small and robust.
    """
    if not (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
    ):
        return {}

    s0 = s.dropna().astype("string")
    if s0.empty:
        return {}

    # cap for speed
    s0 = s0.head(500)

    raw_levels = set(s0.unique().tolist())
    norm = s0.str.strip().str.lower()
    norm_levels = set(norm.unique().tolist())

    # boolean-ish codes
    norm_nonnull = [x for x in norm_levels if x is not None]
    boolish_share = 0.0
    if norm_nonnull:
        boolish_share = float(sum(1 for x in norm_nonnull if str(x) in _BOOL_LEVELS) / max(1, len(norm_nonnull)))

    return {
        "raw_level_count": int(len(raw_levels)),
        "normalized_level_count": int(len(norm_levels)),
        "case_ws_fragmentation": bool(len(raw_levels) > len(norm_levels)),
        "boolish_level_share": float(boolish_share),
    }


def _robust_outlier_rate_iqr(s: pd.Series, k: float = 1.5) -> Optional[float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 20:
        return None
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return None
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return float(((x < lo) | (x > hi)).mean())


def _uniqueness_ratio(s: pd.Series) -> float:
    n = int(s.notna().sum())
    if n == 0:
        return 0.0
    return float(s.nunique(dropna=True) / n)


def _small_corr_summary(df_num: pd.DataFrame, max_cols: int = 10, top_pairs: int = 8) -> List[Dict[str, Any]]:
    """
    Return top abs-correlation pairs among up to max_cols numeric columns.
    """
    cols = df_num.columns.tolist()[:max_cols]
    if len(cols) < 2:
        return []
    c = df_num[cols].corr(numeric_only=True)
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = c.iloc[i, j]
            if pd.isna(v):
                continue
            pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    out = []
    for a, b, v in pairs[:top_pairs]:
        out.append({"a": a, "b": b, "corr": v})
    return out


def build_df_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Efficient, decision-relevant Data Understanding profile.

    Keeps output SMALL for LLM usage but still encodes core CRISP-DM Data Understanding artifacts:
    - Schema inventory (rows/cols/dtypes)
    - Missingness profile (per-col + row-wise concentration)
    - Key/granularity signals (uniqueness ratios; key candidates)
    - Timestamp plausibility + time coverage for candidates
    - Duplicate signals (row duplicates + key-based duplicates for top candidates)
    - Numeric range/outlier candidates (IQR-based rate + basic quantiles)
    - Encoding consistency flags for categoricals (case/ws fragmentation, boolish coding)
    - Uninformative columns flags (mostly-empty, near-constant)
    - Small redundancy signal (top numeric correlations)
    """
    df0 = _safe_sample(df, PROFILE_SAMPLE_ROWS).copy()

    schema = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }

    # --- Missingness ---
    missing_rate: Dict[str, float] = {str(c): float(df0[c].isna().mean()) for c in df0.columns}

    # Row-wise missingness concentration
    row_miss = df0.isna().sum(axis=1)
    row_missing = {
        "p50": int(row_miss.quantile(0.50)) if len(row_miss) else 0,
        "p90": int(row_miss.quantile(0.90)) if len(row_miss) else 0,
        "max": int(row_miss.max()) if len(row_miss) else 0,
    }

    # --- Split columns ---
    num_cols = [c for c in df0.columns if pd.api.types.is_numeric_dtype(df0[c])]
    dt_cols = [c for c in df0.columns if pd.api.types.is_datetime64_any_dtype(df0[c])]
    other_cols = [c for c in df0.columns if c not in set(num_cols) and c not in set(dt_cols)]

    # --- Numeric stats (small) ---
    numeric: Dict[str, Any] = {}
    for c in num_cols:
        s = pd.to_numeric(df0[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        q = s.quantile([0.05, 0.5, 0.95]).to_dict()
        out_rate = _robust_outlier_rate_iqr(s, k=1.5)
        numeric[str(c)] = {
            "count": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else 0.0,
            "min": float(s.min()),
            "p05": float(q.get(0.05, np.nan)),
            "median": float(q.get(0.5, np.nan)),
            "p95": float(q.get(0.95, np.nan)),
            "max": float(s.max()),
            "outlier_rate_iqr": (None if out_rate is None else float(out_rate)),
        }

    # --- Categorical candidates + top counts (limited) ---
    categorical: Dict[str, Any] = {}
    cat_cols: List[str] = []
    for c in other_cols:
        nunique = int(df0[c].nunique(dropna=True))
        if nunique <= 200:
            cat_cols.append(c)
    cat_cols = cat_cols[:PROFILE_MAX_CATEGORICAL_COLS]
    for c in cat_cols:
        categorical[str(c)] = _top_value_counts(df0[c], PROFILE_TOP_K)

    # --- Role candidates / semantic map ---
    roles: Dict[str, Any] = {"id_candidates": [], "time_candidates": [], "measure_candidates": [], "label_candidates": []}

    # ID candidates: name-hint + high uniqueness
    for c in df0.columns:
        cl = str(c).lower()
        if any(h in cl for h in _ID_NAME_HINTS):
            ur = _uniqueness_ratio(df0[c])
            if ur >= 0.8:
                roles["id_candidates"].append({"column": str(c), "uniqueness_ratio": float(ur)})

    # Time candidates: datetime dtype OR parseable with high rate OR name-hint
    for c in df0.columns:
        cl = str(c).lower()
        is_dt, parse_rate = _is_likely_datetime_series(df0[c])
        name_hint = any(h in cl for h in _TIME_NAME_HINTS)
        if is_dt or name_hint:
            cov = {}
            try:
                if pd.api.types.is_datetime64_any_dtype(df0[c]):
                    dt = pd.to_datetime(df0[c], errors="coerce")
                else:
                    dt = _parse_datetime_best_effort(df0[c].dropna().head(500))
                dt2 = pd.to_datetime(dt, errors="coerce").dropna()
                if not dt2.empty:
                    cov = {
                        "min": str(dt2.min()),
                        "max": str(dt2.max()),
                        "n_unique": int(dt2.nunique()),
                    }
            except Exception:
                cov = {}
            roles["time_candidates"].append(
                {
                    "column": str(c),
                    "parse_rate": float(parse_rate),
                    "name_hint": bool(name_hint),
                    "coverage": cov,
                }
            )

    # Measure candidates: numeric with variance
    for c in num_cols:
        s = pd.to_numeric(df0[c], errors="coerce")
        if s.notna().sum() >= 5 and float(s.std(ddof=0)) > 0:
            roles["measure_candidates"].append({"column": str(c)})

    # Label candidates: categorical-ish columns with moderate cardinality
    for c in cat_cols:
        s = df0[c]
        nunique = int(s.nunique(dropna=True))
        if 2 <= nunique <= 200:
            top_share = 0.0
            try:
                vc = s.value_counts(dropna=True)
                top_share = float(vc.iloc[0] / max(1, vc.sum())) if len(vc) else 0.0
            except Exception:
                top_share = 0.0
            roles["label_candidates"].append({"column": str(c), "cardinality": nunique, "top_share": top_share})

    # cap role lists
    roles["id_candidates"] = sorted(roles["id_candidates"], key=lambda x: x["uniqueness_ratio"], reverse=True)[:8]
    roles["time_candidates"] = roles["time_candidates"][:10]
    roles["measure_candidates"] = roles["measure_candidates"][:12]
    roles["label_candidates"] = roles["label_candidates"][:12]

    # --- Duplicate signals ---
    dup_all = float(df0.duplicated().mean()) if len(df0) else 0.0

    key_dups: List[Dict[str, Any]] = []
    key_cols = [x["column"] for x in roles["id_candidates"] if x.get("column") in df0.columns]
    for c in key_cols[:3]:
        try:
            r = float(df0.duplicated(subset=[c]).mean())
            key_dups.append({"keys": [c], "dup_rate": r})
        except Exception:
            continue

    if len(key_cols) >= 2:
        try:
            r = float(df0.duplicated(subset=key_cols[:2]).mean())
            key_dups.append({"keys": key_cols[:2], "dup_rate": r})
        except Exception:
            pass

    duplicates = {
        "duplicate_rate_all_rows": float(dup_all),
        "key_based_duplicates": key_dups[:4],
    }

    # --- Encoding consistency (limited) ---
    encoding: Dict[str, Any] = {}
    for c in cat_cols[: min(len(cat_cols), 12)]:
        encoding[str(c)] = _encoding_consistency_flags(df0[c])

    # --- Uninformative columns ---
    uninformative = {"mostly_empty": [], "near_constant": []}
    for c in df0.columns:
        mr = missing_rate.get(str(c), 0.0)
        if mr >= 0.95:
            uninformative["mostly_empty"].append({"column": str(c), "missing_rate": float(mr)})

        try:
            s = df0[c]
            nunique = int(s.nunique(dropna=True))
            if nunique <= 1:
                uninformative["near_constant"].append({"column": str(c), "reason": "nunique<=1"})
            elif (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_categorical_dtype(s)
            ):
                vc = s.value_counts(dropna=True)
                if len(vc):
                    top_share = float(vc.iloc[0] / max(1, vc.sum()))
                    if top_share >= 0.995:
                        uninformative["near_constant"].append({"column": str(c), "reason": f"top_share={top_share:.3f}"})
        except Exception:
            continue

    uninformative["mostly_empty"] = uninformative["mostly_empty"][:8]
    uninformative["near_constant"] = uninformative["near_constant"][:10]

    # --- Redundancy signals (small) ---
    corr_pairs: List[Dict[str, Any]] = []
    if len(num_cols) >= 2:
        df_num = df0[num_cols].copy()
        corr_pairs = _small_corr_summary(df_num, max_cols=10, top_pairs=8)

    # --- Compact quality flags ---
    quality_flags: List[Dict[str, Any]] = []
    miss_sorted = sorted(missing_rate.items(), key=lambda kv: kv[1], reverse=True)
    for c, r in miss_sorted[:5]:
        if r >= 0.10:
            quality_flags.append({"code": "missingness", "column": c, "value": float(r), "message": "Material missingness."})

    if duplicates["duplicate_rate_all_rows"] >= 0.10:
        quality_flags.append({"code": "duplicates", "value": duplicates["duplicate_rate_all_rows"], "message": "Non-trivial exact duplicates rate."})

    weak_time = [t for t in roles["time_candidates"] if float(t.get("parse_rate", 0.0)) < 0.9 and bool(t.get("name_hint"))]
    for t in weak_time[:3]:
        quality_flags.append(
            {"code": "time_parse_risk", "column": t.get("column"), "value": float(t.get("parse_rate", 0.0)), "message": "Time-like column has low datetime parse rate."}
        )

    for c, e in list(encoding.items())[:10]:
        if isinstance(e, dict) and e.get("case_ws_fragmentation"):
            quality_flags.append({"code": "encoding_fragmentation", "column": c, "message": "Case/whitespace variants likely."})
            break

    quality_flags = quality_flags[:10]

    return {
        "schema": schema,
        "missing_rate": missing_rate,
        "row_missingness": row_missing,
        "numeric": numeric,
        "categorical": categorical,
        "roles": roles,
        "duplicates": duplicates,
        "encoding_consistency": encoding,
        "uninformative": uninformative,
        "correlations": corr_pairs,
        "quality_flags": quality_flags,
    }