# viz/theme.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_PALETTE: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

@dataclass(frozen=True)
class Theme:
    figsize_wide: tuple = (18, 5)
    figsize_square: tuple = (8, 6)
    dpi: int = 140
    grid: bool = True
    palette: List[str] = None

    def __post_init__(self):
        if self.palette is None:
            object.__setattr__(self, "palette", DEFAULT_PALETTE)

DEFAULT_THEME = Theme()