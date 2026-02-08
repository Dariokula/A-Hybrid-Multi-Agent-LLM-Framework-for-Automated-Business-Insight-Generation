# analysis/prescriptive/decision_formulation.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

# Reuse proven driver screening logic (directional, not causal).
from analysis.diagnostic.driver_relationships import (
    _is_id_like,
    _numeric_effects,
    _categorical_lifts,
    run_driver_relationships,
)


# -----------------------------------------------------------------------------
# Prompt heuristics (simple & generic)
# -----------------------------------------------------------------------------
def _infer_objective_direction(prompt: str) -> str:
    """
    Return 'minimize' | 'maximize' | 'unspecified'.

    Intentionally simple and conservative:
    - explicit phrases like "lower is better" win
    - otherwise: common minimize/maximize verbs
    """
    p = (prompt or "").lower()

    if any(k in p for k in ["lower is better", "smaller is better", "reduce", "decrease", "minimize", "lower ", "less "]):
        return "minimize"
    if any(k in p for k in ["higher is better", "increase", "maximize", "grow", "raise", "more "]):
        return "maximize"
    return "unspecified"


def _pick_kpi_from_prompt(df: pd.DataFrame, prompt: str) -> Optional[str]:
    """
    Pick a numeric KPI column.
    Priority:
      1) prompt mentions an existing numeric column name
      2) fallback: first numeric, non-id-like column
    """
    p = (prompt or "").lower()

    # 1) explicit mention
    for c in df.columns:
        if c.lower() in p and pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(c):
            return c

    # 2) fallback
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(c):
            return c
    return None


def _baseline_stats(y: pd.Series) -> Dict[str, Any]:
    yv = pd.to_numeric(y, errors="coerce").dropna()
    if yv.empty:
        return {"n": 0}
    return {
        "n": int(len(yv)),
        "mean": float(yv.mean()),
        "median": float(yv.median()),
        "p10": float(yv.quantile(0.10)),
        "p90": float(yv.quantile(0.90)),
        "min": float(yv.min()),
        "max": float(yv.max()),
    }


def _drivers_to_actions(
    *,
    drivers: List[Dict[str, Any]],
    objective: str,
    effect_field: str,
) -> List[Dict[str, Any]]:
    """
    Convert directional driver numbers into "what direction to push" suggestions.

    objective:
      - minimize KPI: decrease drivers with positive effect; increase drivers with negative effect
      - maximize KPI: increase drivers with positive effect; decrease drivers with negative effect
      - unspecified: direction-neutral wording
    """
    out: List[Dict[str, Any]] = []

    for d in drivers:
        eff = d.get(effect_field)
        try:
            eff = float(eff)
        except Exception:
            continue

        direction = "positive" if eff > 0 else ("negative" if eff < 0 else "zero")

        if objective == "minimize":
            suggestion = "decrease" if eff > 0 else ("increase" if eff < 0 else "no clear direction")
        elif objective == "maximize":
            suggestion = "increase" if eff > 0 else ("decrease" if eff < 0 else "no clear direction")
        else:
            suggestion = "push opposite of KPI increase" if eff != 0 else "no clear direction"

        out.append({**d, "direction": direction, "suggested_change": suggestion})

    return out


def run_decision_formulation(
    *,
    df: pd.DataFrame,
    viz_spec: Dict[str, Any],
    aggregate_ctx: Dict[str, Any],
    prompt: str = "",
) -> Dict[str, Any]:
    """
    Decision formulation = "I want to improve KPI X, what can I do?"

    We screen *directional* drivers (not causal):
      - Numeric: ΔKPI per +1 SD(driver) in KPI units
      - Categorical: lift vs overall mean (shrinkage-stabilized) in KPI units

    Output:
      - One compact driver plot (same as driver_relationships)
      - A small 'context' dict with baseline + top drivers (+ recommended direction)
        for finalize LLM to interpret into decision support.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    y = _pick_kpi_from_prompt(df, prompt)
    if not y or y not in df.columns:
        return {
            "text": "Decision formulation needs a numeric KPI. Mention a numeric column name in the prompt.",
            "figures": [],
            "x": None,
            "y": None,
            "group": None,
            "granularity": None,
        }

    objective = _infer_objective_direction(prompt)

    # keep only rows with KPI
    yv = pd.to_numeric(df[y], errors="coerce")
    df2 = df.loc[yv.notna()].copy()
    if df2.empty:
        return {"text": f"No numeric values available in '{y}'.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    baseline = _baseline_stats(df2[y])

    # same selection logic as driver_relationships
    num_df = _numeric_effects(df2, y=y, max_features=8, min_n=40)
    cat_df = _categorical_lifts(df2, y=y, max_items=10, min_group_n=10, shrinkage_k=20.0, max_levels_per_col=12)

    # reuse the proven plotting output
    fig_res = run_driver_relationships(df=df2, viz_spec={"resolved": {"y": y}}, aggregate_ctx=aggregate_ctx)
    figures = fig_res.get("figures", []) or []

    numeric = []
    if isinstance(num_df, pd.DataFrame) and not num_df.empty:
        numeric = num_df[["feature", "effect", "corr", "n"]].to_dict(orient="records")

    categorical = []
    if isinstance(cat_df, pd.DataFrame) and not cat_df.empty:
        categorical = cat_df[["feature", "value", "lift", "n", "overall_mean"]].to_dict(orient="records")

    numeric_actions = _drivers_to_actions(drivers=numeric, objective=objective, effect_field="effect")
    categorical_actions = _drivers_to_actions(drivers=categorical, objective=objective, effect_field="lift")

    context: Dict[str, Any] = {
        "kpi": y,
        "objective": objective,
        "baseline": baseline,
        "top_numeric_drivers": numeric_actions[:6],
        "top_categorical_drivers": categorical_actions[:8],
        "notes": [
            "Directional screening only (observational). Use as shortlist for investigation/experiments.",
            "Numeric effect: ΔKPI per +1 SD(driver). Categorical lift: deviation vs overall mean (shrinkage-stabilized).",
        ],
    }

    text = (
        f"Decision formulation for KPI '{y}' (objective: {objective}). "
        f"Computed baseline + screened directional drivers (numeric + categorical)."
    )

    return {
        "text": text,
        "figures": figures,
        "x": None,
        "y": y,
        "group": None,
        "granularity": None,
        "context": context,
    }