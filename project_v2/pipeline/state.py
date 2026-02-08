# pipeline/state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class StepHistoryItem:
    step: str
    decision: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineState:
    base_prompt: str
    prompt_additions: List[str] = field(default_factory=list)

    # “global state” / Parameter, die Steps setzen
    params: Dict[str, Any] = field(default_factory=dict)

    # raw decisions per step (planner outputs)
    decisions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    history: List[StepHistoryItem] = field(default_factory=list)

    def composed_prompt(self) -> str:
        parts = [self.base_prompt]
        for add in self.prompt_additions:
            parts.append(f"[USER_FEEDBACK]{add}")
        return "\n".join(parts)