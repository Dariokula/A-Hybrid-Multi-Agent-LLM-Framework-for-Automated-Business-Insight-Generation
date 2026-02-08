# analysis/prescriptive/scenario_evaluation.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from analysis.viz_apply import apply_viz


def _is_id_like(col: str) -> bool:
    cl = col.lower()
    return any(k in cl for k in ["id", "uuid", "guid", "key", "nr", "no.", "number", "pos"])


def _pick_numeric_target(df: pd.DataFrame, prompt: str) -> Optional[str]:
    p = (prompt or "").lower()
    for c in df.columns:
        if c.lower() in p and pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(c):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(c):
            return c
    return None


def _objective_direction(prompt: str) -> str:
    p = (prompt or "").lower()
    if any(w in p for w in ["minimize", "reduce", "lower", "decrease", "shorten", "verringern", "senken", "reduzieren"]):
        return "minimize"
    if any(w in p for w in ["maximize", "increase", "raise", "improve", "steigern", "erhöhen", "maximieren"]):
        return "maximize"
    return "optimize"


def _prep_features(df: pd.DataFrame, y: str, *, max_levels: int = 20) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    d = df.copy()
    yv = pd.to_numeric(d[y], errors="coerce")
    d = d.loc[yv.notna()].copy()
    yv = pd.to_numeric(d[y], errors="coerce").astype(float)

    Xraw = d.drop(columns=[y])

    defaults: Dict[str, Any] = {}
    ranges: Dict[str, Any] = {}
    cat_levels: Dict[str, List[str]] = {}

    for c in list(Xraw.columns):
        if _is_id_like(c):
            Xraw = Xraw.drop(columns=[c])
            continue

        if pd.api.types.is_numeric_dtype(Xraw[c]):
            s = pd.to_numeric(Xraw[c], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            defaults[c] = med
            q1 = float(s.quantile(0.01)) if s.notna().any() else np.nan
            q99 = float(s.quantile(0.99)) if s.notna().any() else np.nan
            ranges[c] = {"p01": q1, "p99": q99}
            Xraw[c] = s.fillna(med)
        else:
            s = Xraw[c].astype("string").str.strip().fillna("(missing)")
            vc = s.value_counts(dropna=False)
            keep = vc.index[:max_levels].astype(str).tolist()
            cat_levels[c] = keep
            mode = str(vc.index[0]) if len(vc) else "(missing)"
            defaults[c] = mode
            Xraw[c] = s.where(s.astype(str).isin(set(keep)), "(other)")

    X = pd.get_dummies(Xraw, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta = {"defaults": defaults, "ranges": ranges, "cat_levels": cat_levels, "raw_columns": list(Xraw.columns), "ohe_columns": list(X.columns)}
    return X, yv, meta


def _build_scenario_row(meta: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    defaults: Dict[str, Any] = meta.get("defaults", {})
    ranges: Dict[str, Any] = meta.get("ranges", {})
    cat_levels: Dict[str, List[str]] = meta.get("cat_levels", {})

    row: Dict[str, Any] = {}
    warns: List[str] = []

    for col, default in defaults.items():
        if col in scenario:
            raw_val = scenario[col]

            if col in ranges:
                try:
                    v = float(str(raw_val).replace(",", ".").replace("days", "").strip())
                    row[col] = v
                    q = ranges.get(col) or {}
                    p01, p99 = q.get("p01"), q.get("p99")
                    if np.isfinite(p01) and np.isfinite(p99) and (v < p01 or v > p99):
                        warns.append(f"Scenario value {col}={v} is outside typical range (p01≈{p01:.3g}, p99≈{p99:.3g}).")
                except Exception:
                    row[col] = float(default)
                    warns.append(f"Could not parse numeric scenario '{col}={raw_val}', used default≈{default:.3g}.")
            else:
                val = str(raw_val).strip().strip('"').strip("'")
                levels = cat_levels.get(col) or []
                if levels and val not in levels:
                    row[col] = "(other)"
                    warns.append(f"Scenario category {col}='{val}' not in top-{len(levels)} levels; mapped to '(other)'.")
                else:
                    row[col] = val if val else default
        else:
            row[col] = default

    return pd.DataFrame([row]), warns


def _predict_distribution(model: RandomForestRegressor, Xrow: pd.DataFrame) -> Tuple[float, float, float]:
    preds = np.array([est.predict(Xrow)[0] for est in model.estimators_], dtype=float)
    return float(np.mean(preds)), float(np.quantile(preds, 0.10)), float(np.quantile(preds, 0.90))


def run_scenario_evaluation(*, df: pd.DataFrame, viz_spec: Dict[str, Any], aggregate_ctx: Dict[str, Any], prompt: str = "", scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Scenario evaluation (no optimization):
    - train RF regression on filtered dataset scope
    - compare baseline (typical values) vs scenario overrides
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    scenario = scenario or {}
    if not isinstance(scenario, dict):
        scenario = {}

    y = _pick_numeric_target(df, prompt)
    if not y:
        return {"text": "Scenario evaluation requires a numeric KPI target (none detected).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    X, yv, meta = _prep_features(df, y, max_levels=20)
    if len(X) < 80 or X.shape[1] < 2:
        return {"text": f"Not enough usable rows/features to model '{y}' for scenario evaluation.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    # train/test split
    rng = np.random.RandomState(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = yv.iloc[tr], yv.iloc[te]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))

    # baseline scenario = defaults
    base_row, base_warn = _build_scenario_row(meta, {})
    scen_row, scen_warn = _build_scenario_row(meta, scenario)

    # align one-hot
    base_ohe = pd.get_dummies(base_row, drop_first=False)
    scen_ohe = pd.get_dummies(scen_row, drop_first=False)
    for c in X.columns:
        if c not in base_ohe.columns:
            base_ohe[c] = 0.0
        if c not in scen_ohe.columns:
            scen_ohe[c] = 0.0
    base_ohe = base_ohe[X.columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scen_ohe = scen_ohe[X.columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    base_mean, base_p10, base_p90 = _predict_distribution(model, base_ohe)
    scen_mean, scen_p10, scen_p90 = _predict_distribution(model, scen_ohe)

    delta = scen_mean - base_mean
    direction = _objective_direction(prompt)
    better = (delta < 0) if direction == "minimize" else (delta > 0) if direction == "maximize" else None

    figs: List[Any] = []
    fig = plt.figure(figsize=(8.2, 4.6))
    ax = fig.add_subplot(111)

    labels = ["Baseline (typical)", "Scenario"]
    means = [base_mean, scen_mean]
    ax.bar(labels, means)

    # error bars: p10/p90 band
    ax.errorbar([0, 1], means, yerr=[[base_mean - base_p10, scen_mean - scen_p10], [base_p90 - base_mean, scen_p90 - scen_mean]], fmt="none", capsize=6)
    local_viz = dict(viz_spec or {})
    local_viz["title"] = local_viz.get("title") or "Scenario evaluation"
    local_viz["subtitle"] = local_viz.get("subtitle") or f"target={y}"
    local_viz.setdefault("x", {})
    local_viz.setdefault("y", {})
    local_viz["x"]["label"] = local_viz["x"].get("label") or "Comparison"
    local_viz["y"]["label"] = local_viz["y"].get("label") or y

    df_for_viz = pd.DataFrame({"label": labels, "mean": means})
    apply_viz(ax=ax, df=df_for_viz, viz_spec=local_viz, x="label", y="mean", group=None)
    ax.text(0.02, 0.95, f"Model holdout: R²≈{r2:.2f}, MAE≈{mae:.3g}", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.02, 0.86, f"Δ (scenario - baseline) ≈ {delta:.3g}", transform=ax.transAxes, va="top", fontsize=9)
    figs.append(fig)

    scen_pairs = ", ".join([f"{k}={v}" for k, v in scenario.items()]) if scenario else "(none provided)"
    lines: List[str] = []
    lines.append(f"Scenario evaluation for KPI '{y}' (RandomForest, no tuning).")
    lines.append(f"- Dataset scope: uses current filtered dataframe (scope filters only).")
    lines.append(f"- Model quality (holdout): R²≈{r2:.2f}, MAE≈{mae:.3g}.")
    lines.append(f"- Baseline prediction (typical values): mean≈{base_mean:.3g} (p10≈{base_p10:.3g}, p90≈{base_p90:.3g})")
    lines.append(f"- Scenario inputs: {scen_pairs}")
    lines.append(f"- Scenario prediction: mean≈{scen_mean:.3g} (p10≈{scen_p10:.3g}, p90≈{scen_p90:.3g})")
    lines.append(f"- Delta: scenario - baseline ≈ {delta:.3g}")

    if better is True:
        lines.append(f"- Interpretation: scenario is **aligned** with the objective ({direction}).")
    elif better is False:
        lines.append(f"- Interpretation: scenario is **not aligned** with the objective ({direction}).")

    warns = [w for w in (scen_warn or []) if isinstance(w, str)]
    if warns:
        lines.append("- Notes: " + " | ".join(warns))

    lines.append("")
    lines.append("Next steps (no optimization):")
    lines.append("- Evaluate multiple scenarios side-by-side (e.g., different avg_type / resource_group settings).")
    lines.append("- Use Candidate ranking to rank feasible options under fixed constraints.")

    return {"text": "\n".join(lines), "figures": figs, "x": None, "y": y, "group": None, "granularity": None}