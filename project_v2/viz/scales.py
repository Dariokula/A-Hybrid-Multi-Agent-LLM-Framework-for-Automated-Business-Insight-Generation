# viz/scales.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ScaleDecision:
    yscale: str = "linear"   # "linear" or "log"
    as_percent: bool = False

def decide_y_scale(series: pd.Series) -> ScaleDecision:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return ScaleDecision()
    # percent heuristic
    if s.min() >= 0 and s.max() <= 1.0:
        return ScaleDecision(yscale="linear", as_percent=True)
    # log heuristic (only if strictly positive and wide range)
    if s.min() > 0 and (s.max() / max(1e-12, s.min())) > 1e3:
        return ScaleDecision(yscale="log", as_percent=False)
    return ScaleDecision()