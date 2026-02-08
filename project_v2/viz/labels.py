# viz/labels.py
from __future__ import annotations
from typing import Optional
from viz.units import UnitResolver

def label_col(col: str, units: UnitResolver) -> str:
    u = units.unit_for(col)
    if u:
        return f"{col} ({u})"
    return col

def title_distribution(col: str) -> str:
    return f"Distribution: {col}"

def title_trend(y: str, gran: Optional[str]) -> str:
    if gran:
        return f"Trend ({gran}): {y}"
    return f"Trend: {y}"