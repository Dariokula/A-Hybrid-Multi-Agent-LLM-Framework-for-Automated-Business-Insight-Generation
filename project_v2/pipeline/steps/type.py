# pipeline/steps/type.py
from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd

from pipeline.state import PipelineState
from agents.factory import AgentFactory
from pipeline.options import TYPE_OPTIONS


def _allowed_for_family(family: str) -> list[str]:
    fam = (family or "").lower().strip()
    return TYPE_OPTIONS.get(fam, [])


def _fallback_type(prompt: str, family: str) -> Dict[str, Any]:
    p = (prompt or "").lower()
    allowed = _allowed_for_family(family)
    t = allowed[0] if allowed else "stats_summary"
    sig: list[str] = []

    def has_any(words: list[str]) -> bool:
        return any(w in p for w in words)

    if family == "descriptive":
        if has_any(["distribution", "histogram", "verteilung", "anteil", "share", "composition"]):
            t = "distribution" if "distribution" in allowed else t
            sig = ["fallback:distribution"]
        elif has_any(["trend", "over time", "time series", "per month", "monthly", "zeit", "pro monat", "pro woche"]):
            t = "trend" if "trend" in allowed else t
            sig = ["fallback:trend"]
        elif has_any(["group", "by", "segment", "compare", "breakdown", "nach", "pro ", "split"]):
            t = "group_compare" if "group_compare" in allowed else t
            sig = ["fallback:group_compare"]
        elif has_any(["correlation", "relationship", "zusammenhang", "korrelation", "crosstab"]):
            t = "relationships" if "relationships" in allowed else t
            sig = ["fallback:relationships"]
        else:
            t = "stats_summary" if "stats_summary" in allowed else t
            sig = ["fallback:stats_summary"]

    elif family == "diagnostic":
        # Heuristic split:
        # - variance_decomposition: "contribution", "pareto", "which categories contribute"
        # - anomaly_explanation: explicit outlier/deviation/delay/late "why high" type questions
        # - driver_relationships: generic driver screening / associations ("drivers of cycle time")
        is_relationship = has_any(["correlation", "relationship", "zusammenhang", "korrelation"])

        is_contribution = has_any(["pareto", "contribution", "variance decomposition", "beitrag", "anteil", "share of"])

        # IMPORTANT: do NOT treat generic "drivers" wording as anomaly intent.
        # Only switch to anomaly_explanation when the prompt clearly speaks about deviations/outliers/delays/late/why-high.
        is_anomaly_intent = has_any([
            "anomaly", "outlier", "abweich", "auffällig",
            "deviation", "delay", "late", "overdue", "slippage",
            "why is", "why are", "warum", "wieso",
            "extreme", "worst", "top overdue", "most overdue",
        ])

        if is_contribution and not is_relationship:
            t = "variance_decomposition" if "variance_decomposition" in allowed else t
            sig = ["fallback:variance_decomposition"]
        elif is_anomaly_intent and not is_relationship:
            t = "anomaly_explanation" if "anomaly_explanation" in allowed else t
            sig = ["fallback:anomaly_explanation"]
        else:
            t = "driver_relationships" if "driver_relationships" in allowed else t
            sig = ["fallback:driver_relationships"]

    elif family == "predictive":
        if has_any(["forecast", "forecasting", "next", "future", "horizon", "predict next"]):
            t = "forecasting" if "forecasting" in allowed else t
            sig = ["fallback:forecasting"]
        elif has_any(["classify", "classification", "which class", "which status", "probability"]):
            t = "classification" if "classification" in allowed else t
            sig = ["fallback:classification"]
        else:
            t = "regression" if "regression" in allowed else t
            sig = ["fallback:regression"]

    elif family == "prescriptive":
        if has_any(["template", "objective", "lever", "constraint", "decision formulation", "entscheidungs", "ziel", "hebel"]):
            t = "decision_formulation" if "decision_formulation" in allowed else t
            sig = ["fallback:decision_formulation"]
        elif has_any(["evaluate scenario", "scenario evaluation", "what happens if", "if we set", "given we set", "unter der bedingung", "was passiert wenn"]):
            t = "scenario_evaluation" if "scenario_evaluation" in allowed else t
            sig = ["fallback:scenario_evaluation"]
        else:
            t = "candidate_ranking" if "candidate_ranking" in allowed else t
            sig = ["fallback:candidate_ranking"]

    return {
        "type": t,
        "confidence": 0.55,
        "signals": sig or ["fallback:default"],
        "rationale": "Heuristic fallback type selection (LLM unavailable).",
        "llm_error": None,
    }


async def step_type(*, df: pd.DataFrame, state: PipelineState, df_profile: Dict[str, Any], agents: AgentFactory):
    prompt = state.composed_prompt()
    family = (state.params.get("family") or "descriptive").lower().strip()

    llm_error: Optional[str] = None
    try:
        out = await agents.run(
            "type",
            prompt=prompt,
            meta={"step": "type", "family": family, "allowed_types": _allowed_for_family(family)},
            df_profile=df_profile,
        )
        if not isinstance(out, dict):
            raise ValueError("type agent returned non-dict output")
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        out = _fallback_type(prompt, family)
        out["llm_error"] = llm_error

    # enforce allowed
    allowed = _allowed_for_family(family)
    if allowed and out.get("type") not in allowed:
        out["type"] = allowed[0]
        out.setdefault("signals", [])
        out["signals"] = list(out["signals"]) + ["type_forced_to_allowed"]
        out["confidence"] = min(float(out.get("confidence") or 0.5), 0.55)

    # --- intent guardrails for diagnostic routing ---
    if family == "diagnostic":
        p = (prompt or "").lower()

        # Strong anomaly explanation cues
        anomaly_cues = [
            "anomaly", "outlier", "abweich", "auffällig",
            "deviation", "delay", "late", "overdue", "slippage",
            "why is", "why are", "warum", "wieso",
            "extreme", "worst", "most overdue",
        ]
        wants_anomaly_explain = any(k in p for k in anomaly_cues)

        # Relationship/association cues (user explicitly wants correlation/association)
        wants_relationship = any(k in p for k in ["correlation", "relationship", "korrelation", "zusammenhang"])

        # Generic "drivers of cycle time / duration" cues (NOT anomaly by itself)
        cycle_time_cues = ["cycle time", "durchlauf", "dlz", "duration", "lead time", "processing time"]
        wants_drivers = any(k in p for k in ["driver", "drivers", "treiber", "influenc", "impact", "factors"])

        # 1) If prompt clearly is about anomaly/deviation/late/outliers and not explicitly correlation: prefer anomaly_explanation
        if (
            wants_anomaly_explain
            and not wants_relationship
            and out.get("type") == "driver_relationships"
            and "anomaly_explanation" in allowed
        ):
            out.setdefault("auto_adjustments", []).append(
                {
                    "reason": "Prompt indicates deviation/delay/outlier explanation; switched to anomaly_explanation (baseline-vs-outlier).",
                    "before": "driver_relationships",
                    "after": "anomaly_explanation",
                }
            )
            out["type"] = "anomaly_explanation"
            try:
                out["confidence"] = min(float(out.get("confidence") or 0.6), 0.88)
            except Exception:
                out["confidence"] = 0.7

        # 2) If prompt is generic driver question about cycle times (and no anomaly cues): prefer driver_relationships
        if (
            wants_drivers
            and any(k in p for k in cycle_time_cues)
            and not wants_anomaly_explain
            and not wants_relationship
            and out.get("type") == "anomaly_explanation"
            and "driver_relationships" in allowed
        ):
            out.setdefault("auto_adjustments", []).append(
                {
                    "reason": "Prompt asks for generic drivers of cycle time (no deviation/outlier wording); switched to driver_relationships.",
                    "before": "anomaly_explanation",
                    "after": "driver_relationships",
                }
            )
            out["type"] = "driver_relationships"
            try:
                out["confidence"] = min(float(out.get("confidence") or 0.6), 0.88)
            except Exception:
                out["confidence"] = 0.7

    state.params["type"] = out["type"]
    state.params["type_confidence"] = out.get("confidence")
    state.params["type_signals"] = out.get("signals", [])
    state.decisions["type"] = out

    meta = {"decision": out, "rationale": out.get("rationale", "")}
    if llm_error:
        meta.setdefault("warnings", []).append(f"type_llm_error: {llm_error}")

    return meta, df