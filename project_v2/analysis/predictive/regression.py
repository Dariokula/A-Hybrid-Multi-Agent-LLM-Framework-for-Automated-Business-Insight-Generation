# analysis/predictive/regression.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Helpers
# -----------------------------
def _is_id_like(col: str) -> bool:
    cl = str(col).lower()
    return any(k in cl for k in ["id", "uuid", "guid", "key", "nr", "no.", "number", "pos"])


def _tufte_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.18)
    ax.grid(False, axis="x")


def _pick_numeric_y(df: pd.DataFrame, resolved_y: Optional[str]) -> Optional[str]:
    if isinstance(resolved_y, str) and resolved_y in df.columns and pd.api.types.is_numeric_dtype(df[resolved_y]):
        return resolved_y
    for c in df.columns:
        if (
            pd.api.types.is_numeric_dtype(df[c])
            and str(c).lower() not in {"n_records", "count", "n"}
            and not _is_id_like(c)
        ):
            return c
    return None


def _prep_features(df: pd.DataFrame, y: str, *, max_levels: int = 20) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Prepare X/y for regression:
    - drop id-like cols
    - numeric: median impute
    - categorical: cap to top levels, map rest to '(other)'
    - one-hot encode
    Returns metadata to build a scenario row later.
    """
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

    meta = {
        "defaults": defaults,
        "ranges": ranges,
        "cat_levels": cat_levels,
        "max_levels": max_levels,
        "raw_columns": list(Xraw.columns),
        "ohe_columns": list(X.columns),
    }
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
                        warns.append(f"{col}={v} outside typical range")
                except Exception:
                    row[col] = float(default)
                    warns.append(f"Could not parse {col}; used default")
            else:
                val = str(raw_val).strip().strip('"').strip("'")
                levels = cat_levels.get(col) or []
                if levels and val not in levels:
                    row[col] = "(other)"
                    warns.append(f"{col} mapped to '(other)'")
                else:
                    row[col] = val if val else default
        else:
            row[col] = default

    return pd.DataFrame([row]), warns


def _aggregate_importances_to_raw(X_columns: List[str], importances: np.ndarray, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Aggregate one-hot importances back to original raw columns.
    """
    imp_map = dict(zip(X_columns, importances))
    ranges: Dict[str, Any] = meta.get("ranges", {}) or {}
    cat_levels: Dict[str, Any] = meta.get("cat_levels", {}) or {}

    rows = []

    # numeric
    for raw in ranges.keys():
        if raw in imp_map:
            rows.append({"raw_feature": raw, "importance": float(imp_map[raw])})

    # categorical
    for raw in cat_levels.keys():
        prefix = f"{raw}_"
        s = 0.0
        any_hit = False
        for c, v in imp_map.items():
            if c.startswith(prefix):
                s += float(v)
                any_hit = True
        if any_hit:
            rows.append({"raw_feature": raw, "importance": float(s)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce").fillna(0.0)
    return out.sort_values("importance", ascending=False)


def _plot_importance_bars(ax: plt.Axes, labels: List[str], values: List[float], *, title: str, slot_count: int = 10) -> None:
    _tufte_axes(ax)
    ax.set_title(title, fontsize=10, pad=10)

    if not values:
        ax.text(0.5, 0.5, "No importances available.", ha="center", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    k = len(values)
    slots = max(slot_count, k)
    x = np.arange(slots)
    heights = np.zeros(slots, dtype=float)
    heights[:k] = np.array(values, dtype=float)

    ax.bar(x, heights, width=0.65)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Importance")


# -----------------------------
# Main
# -----------------------------
def run_regression(
    *,
    df: pd.DataFrame,
    viz_spec: Dict[str, Any],
    aggregate_ctx: Dict[str, Any],
    prompt: str = "",
    scenario: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classification-style layout:
    - Top-left: Actual vs Predicted scatter + trendline + y=x
    - Top-right: Top feature importances (vertical bars, fixed slot width)
    - Bottom row spanning both: textbox with model metrics + scenario prediction

    `prompt` is accepted for pipeline compatibility but not required for logic.
    `scenario` is inference-only (not a filter).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    resolved = (viz_spec or {}).get("resolved") or {}
    y = _pick_numeric_y(df, resolved.get("y"))
    if not y:
        return {"text": "No numeric target found for regression.", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    # scenario may be passed explicitly (from step_analyze) OR embedded in viz_spec
    scen = scenario if isinstance(scenario, dict) else None
    if scen is None:
        scen = (viz_spec or {}).get("scenario")
        if not isinstance(scen, dict):
            scen = (resolved or {}).get("scenario") if isinstance((resolved or {}).get("scenario"), dict) else {}
    scen = scen or {}

    X, yv, meta = _prep_features(df, y, max_levels=20)
    if len(X) < 80 or X.shape[1] < 2:
        return {"text": f"Not enough usable rows/features for regression on '{y}'.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    # deterministic split
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
    y_pred = pd.Series(model.predict(X_test), index=y_test.index, name="pred")

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # importances aggregated to raw
    imp_raw = pd.DataFrame([])
    imp = getattr(model, "feature_importances_", None)
    if imp is not None and len(imp) == X.shape[1]:
        imp_raw = _aggregate_importances_to_raw(list(X.columns), imp, meta).head(10)

    # scenario inference for textbox
    y_hat = None
    scen_warns: List[str] = []
    if scen:
        raw_row, scen_warns = _build_scenario_row(meta, scen)
        raw_row_ohe = pd.get_dummies(raw_row, drop_first=False).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for c in X.columns:
            if c not in raw_row_ohe.columns:
                raw_row_ohe[c] = 0.0
        raw_row_ohe = raw_row_ohe[X.columns]
        y_hat = float(model.predict(raw_row_ohe)[0])

    # ---- Figure: 2 panels + dedicated bottom textbox row ----
    fig = plt.figure(figsize=(13.2, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[12.0, 2.4], width_ratios=[1.15, 0.85])

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, :])
    axT.axis("off")

    # Left: Actual vs Predicted scatter + trendline (trend-like cleanliness)
    df_fit = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred.values})
    df_fit = df_fit.replace([np.inf, -np.inf], np.nan).dropna()

    _tufte_axes(axL)
    axL.scatter(df_fit["Actual"], df_fit["Predicted"], s=45, alpha=0.85)

    # y=x reference
    if len(df_fit) > 0:
        lo = float(min(df_fit["Actual"].min(), df_fit["Predicted"].min()))
        hi = float(max(df_fit["Actual"].max(), df_fit["Predicted"].max()))
        axL.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, alpha=0.7, label="Ideal (y=x)")

    # trendline: Predicted ~ Actual
    if len(df_fit) >= 3:
        x = df_fit["Actual"].to_numpy(dtype=float)
        y_ = df_fit["Predicted"].to_numpy(dtype=float)
        a, b = np.polyfit(x, y_, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 60)
        ys = a * xs + b
        axL.plot(xs, ys, linewidth=1.6, label="Fit (Predicted ~ Actual)")

    # Legend (top-left) for the two lines
    axL.legend(loc="upper left", frameon=False, fontsize=9)

    axL.set_title("Actual vs predicted (holdout)", fontsize=10, pad=10)
    axL.set_xlabel("Actual")
    axL.set_ylabel("Predicted")

    # Right: top feature importances
    labels = imp_raw["raw_feature"].astype(str).tolist() if not imp_raw.empty else []
    vals = imp_raw["importance"].astype(float).tolist() if not imp_raw.empty else []
    _plot_importance_bars(axR, labels=labels, values=vals, title=f"Top feature importances (k={len(vals)})", slot_count=10)

    fig.suptitle("Regression", fontsize=12, y=1.02)

    # Bottom textbox: comprehensive (model + prediction)
    model_info = f"Target: {y} | R²≈{r2:.2f} | MAE≈{mae:.3g} | RMSE≈{rmse:.3g}"
    if scen:
        scen_pairs = ", ".join([f"{k}={v}" for k, v in scen.items()])
        pred_info = f"Scenario prediction: {y}≈{y_hat:.3g} | Inputs: {scen_pairs}"
        notes = ("Notes: " + " | ".join(scen_warns[:2])) if scen_warns else ""
        box_text = model_info + "\n" + pred_info + (("\n" + notes) if notes else "")
    else:
        box_text = model_info + "\nScenario prediction: (no scenario provided)"

    axT.text(
        0.5, 0.5,
        box_text,
        ha="center", va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", alpha=0.55),
        transform=axT.transAxes,
    )

    text = (
        f"Regression model for '{y}'. Holdout: R²≈{r2:.2f}, MAE≈{mae:.3g}. "
        f"Scenario values are used for inference only (not as filters)."
    )
    if y_hat is not None:
        text += f" Scenario prediction: {y}≈{y_hat:.3g}."

    return {"text": text, "figures": [fig], "x": None, "y": y, "group": None, "granularity": None}