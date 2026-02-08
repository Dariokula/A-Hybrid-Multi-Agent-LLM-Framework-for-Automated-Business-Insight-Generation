# pipeline/steps/family.py
from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd

from pipeline.state import PipelineState
from agents.factory import AgentFactory


def _fallback_family(prompt: str) -> Dict[str, Any]:
    p = (prompt or "").lower()
    # deterministic heuristic fallback
    if any(w in p for w in ["why", "root cause", "driver", "treiber", "ursache"]):
        fam = "diagnostic"
        sig = ["fallback:why_or_drivers"]
    elif any(w in p for w in ["forecast", "predict", "prognose", "vorhersage"]):
        fam = "predictive"
        sig = ["fallback:forecast_predict"]
    elif any(w in p for w in ["recommend", "optimize", "what should we do", "empfehlen", "optimier"]):
        fam = "prescriptive"
        sig = ["fallback:recommend_optimize"]
    else:
        fam = "descriptive"
        sig = ["fallback:default_descriptive"]

    return {
        "family": fam,
        "rationale": "LLM unavailable; applied deterministic family routing fallback.",
        "confidence": 0.35,
        "signals": sig,
    }


async def step_family(*, df: pd.DataFrame, state: PipelineState, df_profile: Dict[str, Any], agents: AgentFactory):
    llm_error: Optional[str] = None
    prompt = state.composed_prompt()

    try:
        out = await agents.run("family", prompt=prompt, meta={"step": "family"}, df_profile=df_profile)
        if not isinstance(out, dict):
            raise ValueError("family agent returned non-dict output")
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        out = _fallback_family(prompt)
        out["llm_error"] = llm_error

    state.params["family"] = out["family"]
    state.params["family_confidence"] = out.get("confidence")
    state.params["family_signals"] = out.get("signals", [])
    state.decisions["family"] = out

    meta = {"decision": out, "rationale": out.get("rationale", "")}
    if llm_error:
        meta.setdefault("warnings", []).append(f"family_llm_error: {llm_error}")
    return meta, df