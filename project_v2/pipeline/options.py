# pipeline/options.py

FAMILY_OPTIONS = ["descriptive", "diagnostic", "predictive", "prescriptive"]

# Only the analysis modules you plan to build (per your table).
TYPE_OPTIONS = {
    "descriptive": [
        "stats_summary",   # Global KPI baseline (Statistics summary)
        "distribution",    # Distribution & composition
        "group_compare",   # Segmented aggregates / group comparison
        "trend",           # Trend over time
        "relationships",   # Lightweight relationships (Correlation / Crosstab; screening only)
    ],
    "diagnostic": [
        "variance_decomposition",  # Contribution / variance decomposition (Pareto drivers)
        "anomaly_explanation",     # Anomaly-centered explanation (subset vs baseline)
        "driver_relationships",    # Driver relationships (deeper relationships)
    ],
    "predictive": [
        "forecasting",     # Forecasting (time series)
        "regression",      # Regression (continuous prediction)
        "classification",  # Classification (no risk-scoring)
    ],
    "prescriptive": [
        "decision_formulation",  # Objective / levers / constraints template
        "scenario_evaluation",   # Scenario evaluation (only; no what-if optimization)
        "candidate_ranking",     # Candidate ranking (no optimization)
    ],
}


def all_types() -> list[str]:
    out: list[str] = []
    for v in TYPE_OPTIONS.values():
        out.extend(v)
    # dedupe preserving order
    seen = set()
    res: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res