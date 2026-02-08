# transforms/filters.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd

def validate_filters_against_df(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for f in filters:
        col = f.get("column")
        if col not in df.columns:
            continue
        out.append(f)
    return out

def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    out = df
    for f in filters:
        col, op, val = f["column"], f["op"], f.get("value")
        s = out[col]
        try:
            if op == "==":
                out = out[s == val]
            elif op == "!=":
                out = out[s != val]
            elif op == ">":
                out = out[pd.to_numeric(s, errors="coerce") > val]
            elif op == ">=":
                out = out[pd.to_numeric(s, errors="coerce") >= val]
            elif op == "<":
                out = out[pd.to_numeric(s, errors="coerce") < val]
            elif op == "<=":
                out = out[pd.to_numeric(s, errors="coerce") <= val]
            elif op == "in":
                if not isinstance(val, list):
                    continue
                out = out[s.isin(val)]
            elif op == "contains":
                out = out[s.astype("string").str.contains(str(val), na=False)]
        except Exception:
            # be conservative: if filter fails, ignore it
            continue
    return out.reset_index(drop=True)