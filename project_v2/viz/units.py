# viz/units.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

def _heuristic_unit_from_name(col: str) -> Optional[str]:
    c = (col or "").lower()
    if any(t in c for t in ["_sec", "seconds", "sek"]):
        return "s"
    if any(t in c for t in ["_min", "minutes"]):
        return "min"
    if any(t in c for t in ["_hour", "_hrs", "stunden", "hours", "_h"]):
        return "h"
    if any(t in c for t in ["percent", "pct", "%", "_rate", "quote"]):
        return "%"
    if any(t in c for t in ["eur", "euro", "revenue", "umsatz", "price"]):
        return "€"
    return None

@dataclass
class UnitResolver:
    """
    Simple resolver: overrides > domain-knowledge mapping > heuristics.
    """
    overrides: Dict[str, str] = None
    dk_units: Dict[str, str] = None

    def __post_init__(self):
        if self.overrides is None:
            self.overrides = {}
        if self.dk_units is None:
            self.dk_units = {}

    def unit_for(self, col: str) -> Optional[str]:
        if col in self.overrides:
            return self.overrides[col]
        if col in self.dk_units:
            return self.dk_units[col]
        return _heuristic_unit_from_name(col)

    @staticmethod
    def from_domain_knowledge(dk: Any) -> "UnitResolver":
        """
        Placeholder: if later DK selector returns structured units, map them here.
        Currently returns empty DK mapping.
        """
        return UnitResolver(overrides={}, dk_units={})