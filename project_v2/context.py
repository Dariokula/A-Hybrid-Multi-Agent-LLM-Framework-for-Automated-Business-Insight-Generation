# context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd

@dataclass
class AnalysisContext:
    df: pd.DataFrame
    df_profile: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)