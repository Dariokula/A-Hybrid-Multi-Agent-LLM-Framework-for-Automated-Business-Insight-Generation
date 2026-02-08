# pipeline/steps/verify.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from domain_knowledge.selector import select_domain_knowledge


_TIME_NAME_HINTS = [
    "dlz",
    "cycle",
    "lead",
    "durchlauf",
    "dauer",
    "time",
    "stunden",
    "hours",
    "mins",
    "minutes",
    "seconds",
    "sek",
]


def _extract_counts_df_from_state(state) -> Optional[pd.DataFrame]:
    """Fetch aggregation cockpit counts table if present."""
    try:
        cockpit = getattr(state, "params", {}).get("aggregation_cockpit") or {}
        cdf = cockpit.get("counts_df")
        if isinstance(cdf, pd.DataFrame):
            return cdf
    except Exception:
        pass
    return None


def _select_dk(prompt: str, state, top_k: int = 3) -> Dict[str, Any]:
    """Compatibility wrapper for different selector signatures."""
    try:
        return select_domain_knowledge(prompt, state=state, top_k=top_k)  # type: ignore[arg-type]
    except TypeError:
        try:
            return select_domain_knowledge(prompt, top_k=top_k)  # type: ignore[call-arg]
        except TypeError:
            try:
                return select_domain_knowledge(prompt)  # type: ignore[call-arg]
            except Exception:
                return {"snippets": [], "debug": {"reason": "dk_selector_failed"}}


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _time_col(df: pd.DataFrame) -> Optional[str]:
    if "time_bucket" in df.columns:
        return "time_bucket"
    for c in df.columns:
        cl = str(c).lower()
        if "date" in cl or "time" in cl:
            return c
    return None


def _robust_z_outliers(x: pd.Series, zthr: float = 3.5) -> pd.Series:
    x2 = pd.to_numeric(x, errors="coerce")
    med = x2.median()
    mad = (x2 - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series([False] * len(x2), index=x2.index)
    rz = 0.6745 * (x2 - med) / mad
    return rz.abs() > zthr


def _basic_stats(s: pd.Series) -> Dict[str, Any]:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return {"n": 0}
    q1 = float(s2.quantile(0.25))
    q3 = float(s2.quantile(0.75))
    iqr = float(q3 - q1)
    return {
        "n": int(len(s2)),
        "mean": float(s2.mean()),
        "median": float(s2.median()),
        "std": float(s2.std(ddof=0)) if len(s2) > 1 else 0.0,
        "min": float(s2.min()),
        "p05": float(s2.quantile(0.05)),
        "p95": float(s2.quantile(0.95)),
        "p99": float(s2.quantile(0.99)) if len(s2) >= 20 else float(s2.max()),
        "max": float(s2.max()),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
    }


def _slim_findings(findings: Any, max_items: int = 6) -> List[Dict[str, Any]]:
    if not isinstance(findings, list):
        return []
    out: List[Dict[str, Any]] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        out.append(
            {
                "code": f.get("code"),
                "severity": f.get("severity"),
                "message": (str(f.get("message") or "")[:240] if f.get("message") is not None else ""),
                "evidence": f.get("evidence", None) if isinstance(f.get("evidence"), (str, int, float, bool)) else None,
            }
        )
        if len(out) >= max_items:
            break
    return out


def _slim_stats(stats: Any, keep_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    keep_cols = keep_cols or []
    out: Dict[str, Any] = {}
    for k in keep_cols:
        v = stats.get(k)
        if isinstance(v, dict):
            out[k] = {kk: v.get(kk) for kk in ["n", "mean", "median", "min", "max", "p05", "p95"] if kk in v}
    return out


def _slim_outliers(anomalies: Any, max_items: int = 3) -> Dict[str, Any]:
    if not isinstance(anomalies, dict):
        return {}
    outliers = anomalies.get("outliers")
    if not isinstance(outliers, list) or not outliers:
        return {}
    slim: List[Dict[str, Any]] = []
    for o in outliers:
        if not isinstance(o, dict):
            continue
        slim.append({"x": o.get("x"), "y": o.get("y")})
        if len(slim) >= max_items:
            break
    return {"outliers": slim}


def _looks_aggregated_table(df: pd.DataFrame) -> bool:
    """Heuristic: table likely aggregated if it contains support counts or many mean_/sum_ cols."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    if any(c in df.columns for c in ["n_records", "count", "n", "rows"]):
        return True
    prefixes = ("mean_", "median_", "sum_", "min_", "max_", "avg_")
    pref_cols = [c for c in df.columns if str(c).lower().startswith(prefixes)]
    return (len(pref_cols) / max(1, len(df.columns))) >= 0.25


def _find_support_series(df: pd.DataFrame, counts_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Return per-row support as a Series (n_records) and a source tag:
      - 'df:<col>' if found directly in df
      - 'cockpit_join' if derived by joining against counts_df
      - (None, None) if not found
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None

    # 1) direct support columns in df
    support_cols = ["n_records", "count", "n", "rows", "row_count", "records"]
    for c in support_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s, f"df:{c}"

    # 2) join against cockpit counts_df if available
    if counts_df is None or not isinstance(counts_df, pd.DataFrame) or counts_df.empty:
        return None, None
    if not any(c in counts_df.columns for c in ["n_records", "count", "n"]):
        return None, None

    cdf = counts_df.copy()
    # normalize support col name in cockpit
    if "n_records" not in cdf.columns:
        for alt in ["count", "n"]:
            if alt in cdf.columns:
                cdf["n_records"] = pd.to_numeric(cdf[alt], errors="coerce")
                break

    if "n_records" not in cdf.columns:
        return None, None

    df0 = df.copy()

    join_keys: List[str] = []

    # time_bucket is the canonical key if present
    if "time_bucket" in df0.columns and "time_bucket" in cdf.columns:
        df0["_tb_tmp"] = pd.to_datetime(df0["time_bucket"], errors="coerce")
        cdf["_tb_tmp"] = pd.to_datetime(cdf["time_bucket"], errors="coerce")
        join_keys.append("_tb_tmp")

    # add shared low-cardinality categorical keys (avoid joining on high-cardinality IDs)
    for c in df0.columns:
        if c in {"_tb_tmp"}:
            continue
        if c in cdf.columns and c not in {"n_records", "count", "n"}:
            try:
                nun = int(df0[c].nunique(dropna=True))
                if nun <= 50:
                    join_keys.append(c)
            except Exception:
                continue

    join_keys = [k for k in join_keys if k in df0.columns and k in cdf.columns]
    join_keys = list(dict.fromkeys(join_keys))

    if not join_keys:
        # last resort: if cockpit is only per time_bucket
        if "_tb_tmp" in df0.columns and "_tb_tmp" in cdf.columns:
            join_keys = ["_tb_tmp"]
        else:
            return None, None

    merged = df0[join_keys].merge(cdf[join_keys + ["n_records"]], on=join_keys, how="left")

    # cleanup temp column
    if "_tb_tmp" in df0.columns:
        df0.drop(columns=["_tb_tmp"], inplace=True, errors="ignore")

    s = pd.to_numeric(merged["n_records"], errors="coerce")
    if s.notna().any():
        return s, "cockpit_join"
    return None, None


def _deterministic_verify(
    *,
    df: pd.DataFrame,
    prompt: str,
    params: Dict[str, Any],
    signature: Dict[str, Any],
    thresholds: Dict[str, Any],
    counts_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    anomalies: Dict[str, Any] = {}

    df0 = df.copy()

    # Guardrail: diagnostic explanation tasks should be on row-level input
    fam = str(params.get("family") or "").lower().strip()
    typ = str(params.get("type") or "").lower().strip()
    if fam == "diagnostic" and typ in {"driver_relationships", "variance_decomposition", "anomaly_explanation"}:
        if _looks_aggregated_table(df0):
            findings.append(
                {
                    "code": "aggregated_input_for_diagnostic",
                    "severity": "warn",
                    "message": "Input dataframe looks aggregated (e.g., mean_* columns or support counts). "
                               "Driver/anomaly explanations are typically more reliable on row-level data.",
                    "evidence": {"type": typ, "hint": "prefer disabling aggregate() for these diagnostic types"},
                }
            )

    x = signature.get("x") if isinstance(signature, dict) else None
    y = signature.get("y") if isinstance(signature, dict) else None
    if not x:
        x = _time_col(df0)
    if not y:
        ncols = _numeric_cols(df0)
        y = next((c for c in ncols if c != x), None)

    # time duplicates check
    if x and x in df0.columns:
        dup_rate = float(df0[x].duplicated().mean())
        if dup_rate > thresholds.get("dup_rate_warn", 0.15):
            findings.append(
                {
                    "code": "time_duplicates",
                    "severity": "warn",
                    "message": f"Time column '{x}' has duplicated buckets (dup_rate={dup_rate:.3f}).",
                    "evidence": {"x": x, "dup_rate": dup_rate},
                }
            )

    # stats + outliers for y
    if y and y in df0.columns:
        stats[y] = _basic_stats(df0[y])

        if x and x in df0.columns:
            out_mask = _robust_z_outliers(df0[y], thresholds.get("robust_z_outlier", 3.5))
            if bool(out_mask.any()):
                idxs = df0.index[out_mask].tolist()[:5]
                rows: List[Dict[str, Any]] = []
                for i in idxs:
                    rows.append(
                        {
                            "index": int(i),
                            "y": float(pd.to_numeric(df0.loc[i, y], errors="coerce")),
                            "x": str(df0.loc[i, x]),
                        }
                    )
                anomalies["outliers"] = rows
                findings.append(
                    {
                        "code": "series_outliers",
                        "severity": "warn",
                        "message": f"Detected outlier bucket(s) in '{y}' using robust z-score.",
                        "evidence": {
                            "y": y,
                            "robust_z_threshold": thresholds.get("robust_z_outlier", 3.5),
                            "examples": rows[:3],
                        },
                    }
                )

        # Low support warning (prefer df support col; else use cockpit join)
        nrec_series, support_source = _find_support_series(df0, counts_df)
        if nrec_series is not None:
            nrec = nrec_series.fillna(0)
            thr = thresholds.get("min_bucket_support_warn", 5)
            low_fraction = float((nrec < thr).mean())
            if low_fraction > 0:
                findings.append(
                    {
                        "code": "low_bucket_support",
                        "severity": "warn",
                        "message": f"Some buckets have low support (n_records < {thr}). Averages may be unstable.",
                        "evidence": {"low_bucket_fraction": low_fraction, "min_bucket_support_warn": thr, "source": support_source},
                    }
                )
                stats["n_records"] = _basic_stats(nrec)

    # Prompt/params consistency checks (lightweight)
    p = (prompt or "").lower()
    filters = params.get("filters") or []
    if "finished" in p and "part" in p:
        has_status_filter = any(
            (isinstance(f, dict) and str(f.get("column", "")).lower() == "status") for f in filters
        )
        if not has_status_filter:
            findings.append(
                {
                    "code": "missing_finished_filter",
                    "severity": "warn",
                    "message": "Prompt requests 'only finished parts' but no explicit status filter was detected in params.",
                    "evidence": {"filters": filters[:5]},
                }
            )

    severities = [f.get("severity") for f in findings]
    if "block" in severities:
        status_prelim = "block"
    elif "warn" in severities:
        status_prelim = "warn"
    else:
        status_prelim = "ok"

    return {
        "status_prelim": status_prelim,
        "findings": findings,
        "stats": stats,
        "anomalies": anomalies,
        "signature": signature or {"x": x, "y": y},
    }


def _build_verify_meta_for_llm(
    *,
    prompt: str,
    params: Dict[str, Any],
    df: pd.DataFrame,
    df_profile: Dict[str, Any],
    det: Dict[str, Any],
    dk: Dict[str, Any],
) -> Dict[str, Any]:
    signature = det.get("signature") or {}
    y = signature.get("y")

    det_findings = det.get("findings", [])
    det_stats = det.get("stats", {})
    det_anom = det.get("anomalies", {})

    keep_cols: List[str] = []
    if isinstance(y, str) and y:
        keep_cols.append(y)
    if "n_records" in det_stats:
        keep_cols.append("n_records")

    slim_det = {
        "status_prelim": det.get("status_prelim"),
        "signature": signature,
        "findings": _slim_findings(det_findings, max_items=6),
        "stats": _slim_stats(det_stats, keep_cols=keep_cols),
        "anomalies": _slim_outliers(det_anom, max_items=3),
    }

    dk_snips = []
    if isinstance(dk, dict) and isinstance(dk.get("snippets"), list):
        for sn in dk.get("snippets")[:2]:
            if not isinstance(sn, dict):
                continue
            dk_snips.append(
                {
                    "source": sn.get("source"),
                    "type": sn.get("type"),
                    "score": sn.get("score"),
                    "text": str(sn.get("text") or "")[:400],
                }
            )

    schema = (df_profile or {}).get("schema") or {}
    small_profile = {
        "schema": {
            "rows": schema.get("rows"),
            "cols": schema.get("cols"),
            "dtypes": (schema.get("dtypes") or {}),
        }
    }

    return {
        "step": "verify",
        "family": params.get("family"),
        "type": params.get("type"),
        "filters": params.get("filters", []),
        "columns": params.get("columns", list(df.columns)),
        "prepare_actions": params.get("prepare_actions", []),
        "aggregate": params.get("aggregate") or {},
        "viz_spec": params.get("viz_spec") or {},
        "analysis_signature": signature,
        "deterministic": slim_det,
        "domain_knowledge": {"snippets": dk_snips, "debug": (dk.get("debug") if isinstance(dk, dict) else {})},
        "output_constraints": {
            "max_issues": 6,
            "max_next_actions": 4,
            "max_user_warning_chars": 320,
        },
        "df_profile_small": small_profile,
        "prompt_hint": (prompt[:400] if prompt else ""),
    }


async def step_verify(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents):
    prompt = state.composed_prompt()
    params = state.params or {}
    signature = (params.get("analysis_signature") or {}) if isinstance(params.get("analysis_signature"), dict) else {}

    thresholds = {
        "dup_rate_warn": 0.15,
        "robust_z_outlier": 3.5,
        "min_bucket_support_warn": 5,
    }

    det = _deterministic_verify(
        df=df,
        prompt=prompt,
        params=params,
        signature=signature,
        thresholds=thresholds,
        counts_df=_extract_counts_df_from_state(state),
    )

    dk = _select_dk(prompt, state=state, top_k=3)
    llm_error: Optional[str] = None

    meta_for_llm = _build_verify_meta_for_llm(
        prompt=prompt, params=params, df=df, df_profile=df_profile, det=det, dk=dk
    )

    try:
        out_llm = await agents.run(
            "verify",
            prompt=prompt,
            meta=meta_for_llm,
            df_profile=df_profile,
        )
        if not isinstance(out_llm, dict):
            raise ValueError("Verify agent returned non-dict output")
    except Exception as e:
        # IMPORTANT: do not fail the pipeline just because LLM verify failed.
        llm_error = f"{type(e).__name__}: {e}"
        out_llm = {
            "status": "ok",
            "issues": [],
            "user_warning": "Automated verification ran in deterministic fallback mode (LLM verification unavailable).",
            "next_actions": [],
            "confidence": 0.4,
            "rationale": "LLM verify call failed; using deterministic checks only.",
        }

    det_status = det.get("status_prelim", "ok")
    llm_status = out_llm.get("status", "ok")

    order = {"ok": 0, "warn": 1, "block": 2}
    status = det_status if order.get(det_status, 0) >= order.get(llm_status, 0) else llm_status

    decision = {
        "status": status,
        "status_prelim": det_status,
        "llm_status": llm_status,
        "llm_error": llm_error,
        "deterministic": det,
        "deterministic_slim_sent": (meta_for_llm.get("deterministic") or {}),
        "issues": out_llm.get("issues", []),
        "user_warning": out_llm.get("user_warning", ""),
        "next_actions": out_llm.get("next_actions", []),
        "confidence": out_llm.get("confidence", 0.5),
        "rationale": out_llm.get("rationale", ""),
    }

    meta = {
        "decision": decision,
        "rationale": decision.get("rationale", ""),
        "df_delta": None,
    }
    return meta, df