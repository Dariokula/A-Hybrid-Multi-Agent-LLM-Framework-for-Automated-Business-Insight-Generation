# pipeline/registry.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class StepSpec:
    name: str
    agent_id: Optional[str] = None          # which agent (if any)
    options: Dict[str, Any] = field(default_factory=dict)  # step knobs/options shown to user
    instruction_key: Optional[str] = None   # key for instructions (often same as agent_id)

STEP_SPECS: Dict[str, StepSpec] = {
    "prompt": StepSpec(
        name="prompt",
        agent_id=None,
        options={"purpose": "store initial prompt"},
        instruction_key=None,
    ),
    "family": StepSpec(
        name="family",
        agent_id="family",
        options={"allowed_families": ["descriptive","diagnostic","predictive","prescriptive"]},
        instruction_key="family",
    ),
    "type": StepSpec(
        name="type",
        agent_id="type",
        options={"note": "type is selected within the chosen family"},
        instruction_key="type",
    ),
    "filters": StepSpec(
        name="filters",
        agent_id="filters",
        options={
            "allowed_ops": ["==","!=",">",">=","<","<=","in","contains"],
            "safety": "conservative filtering; if uncertain, return empty",
        },
        instruction_key="filters",
    ),
    "columns": StepSpec(
        name="columns",
        agent_id="columns",
        options={
            "max_columns_hint": 12,
            "must_keep": "filter columns (if any) should be kept (later improvement)",
        },
        instruction_key="columns",
    ),
    "prepare": StepSpec(
        name="prepare",
        agent_id="prepare",
        options={
            "allowed_actions": [
                "drop_duplicates",
                "strip_strings",
                "coerce_numeric",
                "parse_datetimes",
                "impute_numeric_median",
                "impute_categorical_mode",
                "drop_rows_missing_required",
                "clip_outliers_iqr",
            ],
            "max_row_drop_fraction": 0.10,  # safety guardrail
            "outlier_iqr_k": 1.5,
        },
        instruction_key="prepare",
    ),
    "aggregate": StepSpec(
        name="aggregate",
        agent_id="aggregate",
        options={
            "allowed_granularities": ["day","week","month","quarter","year"],
            "allowed_aggs": ["count","sum","mean","median","min","max"],
        },
        instruction_key="aggregate",
    ),
    "viz": StepSpec(
        name="viz",
        agent_id="viz",
        instruction_key="viz",
        options={
            "allowed_y_scales": ["linear", "log"],
            "legend_locs": ["best","upper right","upper left","lower right","lower left","center left","center right"],
            "date_formats": ["auto","%Y-%m","%Y-%m-%d","%d.%m.%Y"],
        },
    ),
    "analyze": StepSpec(
        name="analyze",
        agent_id=None,
        options={"note": "routes to analysis/<family>/... later"},
        instruction_key=None,
    ),
    "verify": StepSpec(
        name="verify",
        agent_id="verify",
        options={
            "note": "validates analysis output (deterministic + LLM audit) before presentation",
        },
        instruction_key="verify",
    ),
    "finalize": StepSpec(
        name="finalize",
        agent_id="finalize",
        options={
            "note": "builds narrative & highlights based on trace/stats/verify/domain knowledge",
        },
        instruction_key="finalize",
    ),
}