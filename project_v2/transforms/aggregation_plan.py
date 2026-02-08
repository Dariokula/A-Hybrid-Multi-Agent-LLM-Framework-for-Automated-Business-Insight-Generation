# transforms/aggregation_plan.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd

def apply_aggregation(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Very conservative aggregation:
    - optional time grouping
    - optional groupby columns
    - metrics list (agg on numeric columns)
    If plan empty or invalid -> return df unchanged.
    """
    info: Dict[str, Any] = {"used": False, "groupby": {}, "time_grouping": None, "metrics": []}

    if not isinstance(plan, dict):
        return df, info

    groupby_cols = plan.get("groupby_columns") or []
    if not isinstance(groupby_cols, list):
        groupby_cols = []

    tg = plan.get("time_grouping")
    if tg and isinstance(tg, dict):
        col = tg.get("column")
        freq = tg.get("freq", "W")
        out_col = tg.get("output_column") or f"{col}_{freq}"
        if col in df.columns:
            # try parse datetime
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                df = df.copy()
                df[out_col] = dt.dt.to_period(freq).dt.start_time
                groupby_cols = [out_col] + [c for c in groupby_cols if c != out_col]
                info["time_grouping"] = {"column": col, "freq": freq, "output_column": out_col}

    groupby_cols = [c for c in groupby_cols if c in df.columns]
    metrics = plan.get("metrics") or []
    if not isinstance(metrics, list):
        metrics = []

    # If nothing to do, return unchanged
    if not groupby_cols or not metrics:
        return df.reset_index(drop=True), info

    agg_dict = {}
    rename = {}
    for m in metrics:
        if not isinstance(m, dict):
            continue
        col = m.get("column")
        agg = m.get("agg")
        alias = m.get("as") or f"{agg}_{col}"
        if col not in df.columns or not agg:
            continue
        agg = str(agg)
        if agg not in {"sum","mean","count","median","min","max"}:
            continue
        agg_dict.setdefault(col, []).append(agg)
        rename[(col, agg)] = alias

    if not agg_dict:
        return df.reset_index(drop=True), info

    g = df.groupby(groupby_cols, dropna=False).agg(agg_dict)
    # flatten multiindex columns
    g.columns = [rename.get((c, a), f"{a}_{c}") for c, a in g.columns]
    g = g.reset_index()

    info["used"] = True
    info["groupby"] = {"used": groupby_cols}
    info["metrics"] = [{"column": k, "agg": a} for k, aggs in agg_dict.items() for a in aggs]
    return g.reset_index(drop=True), info