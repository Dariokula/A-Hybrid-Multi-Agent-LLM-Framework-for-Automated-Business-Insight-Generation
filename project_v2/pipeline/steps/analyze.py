# pipeline/steps/analyze.py
from __future__ import annotations
from typing import Any, Dict
import pandas as pd

# Descriptive
from analysis.descriptive.distribution import run_distribution
from analysis.descriptive.stats_summary import run_stats_summary
from analysis.descriptive.group_compare import run_group_compare
from analysis.descriptive.relationships import run_relationships
from analysis.descriptive.trend import run_trend

# Diagnostic
from analysis.diagnostic.variance_decomposition import run_variance_decomposition
from analysis.diagnostic.anomaly_explanation import run_anomaly_explanation
from analysis.diagnostic.driver_relationships import run_driver_relationships

# Predictive
from analysis.predictive.forecasting import run_forecasting
from analysis.predictive.regression import run_regression
from analysis.predictive.classification import run_classification

# Prescriptive
from analysis.prescriptive.decision_formulation import run_decision_formulation
from analysis.prescriptive.scenario_evaluation import run_scenario_evaluation
from analysis.prescriptive.candidate_ranking import run_candidate_ranking


def _infer_anomaly_mode_from_prompt(prompt: str) -> str:
    p = (prompt or "").lower()

    high_markers = [
        "late", "delay", "delays", "behind", "overdue", "exceed", "exceeds", "largest", "highest", "worst",
        "extreme late", "most extreme late", "most late", "spät", "verspät", "verzug", "zu spät", "am spätesten",
        "high outlier", "upper tail", "top outlier",
    ]
    low_markers = [
        "early", "ahead", "negative", "smallest", "lowest", "best", "shortest", "fastest",
        "früh", "zu früh", "am frühesten", "low outlier", "lower tail", "bottom outlier",
    ]

    has_high = any(m in p for m in high_markers)
    has_low = any(m in p for m in low_markers)

    if has_high and not has_low:
        return "high"
    if has_low and not has_high:
        return "low"
    return "both"


async def step_analyze(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents):
    fam = (state.params.get("family", "descriptive") or "").strip().lower()
    typ = (state.params.get("type", "distribution") or "").strip().lower()

    viz_spec = state.params.get("viz_spec") or {}
    aggregate_ctx = state.params.get("aggregate") or {}
    prompt_text = state.composed_prompt() or ""

    scenario = state.params.get("scenario") or {}
    if not isinstance(scenario, dict):
        scenario = {}

    if fam == "descriptive" and typ == "stats_summary":
        res = run_stats_summary(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx)
    elif fam == "descriptive" and typ == "distribution":
        res = run_distribution(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx)
    elif fam == "descriptive" and typ == "group_compare":
        res = run_group_compare(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx)
    elif fam == "descriptive" and typ == "relationships":
        res = run_relationships(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx)
    elif fam == "descriptive" and typ == "trend":
        res = run_trend(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx)

    elif fam == "diagnostic" and typ == "variance_decomposition":
        res = run_variance_decomposition(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx)

    elif fam == "diagnostic" and typ == "anomaly_explanation":
        anomaly_mode = _infer_anomaly_mode_from_prompt(prompt_text)
        state.params["anomaly_mode"] = anomaly_mode
        if isinstance(viz_spec, dict):
            viz_spec = dict(viz_spec)
            viz_spec["anomaly_mode"] = anomaly_mode
            state.params["viz_spec"] = viz_spec
        res = run_anomaly_explanation(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx, prompt=prompt_text)

    elif fam == "diagnostic" and typ == "driver_relationships":
        res = run_driver_relationships(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx)

    elif fam == "predictive" and typ == "forecasting":
        # ✅ pass prompt so horizon can be derived from user request
        res = run_forecasting(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx, prompt=prompt_text)

    elif fam == "predictive" and typ == "regression":
        res = run_regression(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx, prompt=prompt_text, scenario=scenario)
    elif fam == "predictive" and typ == "classification":
        res = run_classification(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx, prompt=prompt_text, scenario=scenario)

    elif fam == "prescriptive" and typ == "decision_formulation":
        res = run_decision_formulation(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx, prompt=prompt_text)
    elif fam == "prescriptive" and typ == "scenario_evaluation":
        res = run_scenario_evaluation(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx, prompt=prompt_text, scenario=scenario)
    elif fam == "prescriptive" and typ == "candidate_ranking":
        res = run_candidate_ranking(df=df, viz_spec=viz_spec, aggregate_ctx=aggregate_ctx, prompt=prompt_text, scenario=scenario)

    else:
        res = {
            "text": f"No analysis module implemented yet for family='{fam}', type='{typ}'.",
            "figures": [],
            "x": None,
            "y": None,
            "group": None,
            "granularity": None,
        }

    figures = res.get("figures", []) or []
    text = res.get("text", "") or ""

    analysis_context = None
    if isinstance(res, dict) and isinstance(res.get("context"), dict):
        analysis_context = res.get("context")

    signature = {
        "x": res.get("x"),
        "y": res.get("y"),
        "group": res.get("group"),
        "granularity": res.get("granularity"),
    }

    state.params["analysis_signature"] = signature
    state.params["analysis_text"] = text
    if analysis_context is not None:
        state.params["analysis_context"] = analysis_context
    else:
        state.params.pop("analysis_context", None)
    state.decisions["analyze"] = {"family": fam, "type": typ}

    meta = {
        "decision": state.decisions["analyze"],
        "rationale": "Analyze executed.",
        "artifacts": {"figures": figures},
        "final_output": {"text": text},
        "text": text,
        "df_delta": None,
    }
    return meta, df