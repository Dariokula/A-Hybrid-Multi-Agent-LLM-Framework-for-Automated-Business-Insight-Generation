# pipeline/context_builder.py
from __future__ import annotations

from typing import Any, Dict
from pipeline.registry import StepSpec


def _slim_df_profile(df_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only what's decision-relevant for LLM + review.
    Prevent oversized payloads (local runs + OpenRouter free providers).
    """
    schema = (df_profile.get("schema") or {})
    return {
        "schema": schema,
        "missing_rate": (df_profile.get("missing_rate") or {}),
        "row_missingness": (df_profile.get("row_missingness") or {}),
        "numeric": (df_profile.get("numeric") or {}),
        "categorical": (df_profile.get("categorical") or {}),
        "roles": (df_profile.get("roles") or {}),
        "duplicates": (df_profile.get("duplicates") or {}),
        "encoding_consistency": (df_profile.get("encoding_consistency") or {}),
        "uninformative": (df_profile.get("uninformative") or {}),
        "correlations": (df_profile.get("correlations") or []),
        "quality_flags": (df_profile.get("quality_flags") or []),
    }


def build_step_input(
    *,
    step: str,
    spec: StepSpec,
    composed_prompt: str,
    params: Dict[str, Any],
    df_profile: Dict[str, Any],
    instructions: str,
    domain_knowledge: Dict[str, Any],
    step_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Step input shown in your run report / review UI.

    IMPORTANT efficiency choice:
    - Do NOT attach the full df_profile if it grows.
    - Attach a slim df_profile that contains the key Data Understanding artifacts.
    """
    dfp = _slim_df_profile(df_profile)
    schema = (dfp.get("schema") or {})
    return {
        "step": step,
        "prompt": composed_prompt,
        "chosen_params_so_far": dict(params),
        "df_schema": schema,
        "df_profile": dfp,  # slim by design
        "step_options": step_options if step_options is not None else (spec.options or {}),
        "step_instructions": instructions or "",
        "domain_knowledge": domain_knowledge or {},
    }