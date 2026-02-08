# transforms/columns.py
from __future__ import annotations
from typing import List
import pandas as pd
from config import MAX_COLUMNS_DEFAULT

def apply_column_selection(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if not columns:
        # conservative fallback: keep first N columns
        cols = list(df.columns)[:MAX_COLUMNS_DEFAULT]
        return df.loc[:, cols].reset_index(drop=True)
    cols = [c for c in columns if c in df.columns]
    if not cols:
        cols = list(df.columns)[:MAX_COLUMNS_DEFAULT]
    return df.loc[:, cols].reset_index(drop=True)