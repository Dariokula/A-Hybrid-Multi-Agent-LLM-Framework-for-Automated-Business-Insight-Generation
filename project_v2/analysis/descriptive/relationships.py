# analysis/descriptive/relationships.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.viz_apply import apply_viz, infer_inlier_bounds


def _is_categorical_like(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    )


def _is_id_like(col: str) -> bool:
    cl = col.lower()
    return any(k in cl for k in ["id", "uuid", "guid", "key", "nr", "no.", "number", "pos"])


def _pick_numeric_cols(df: pd.DataFrame, max_cols: int = 10) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if _is_id_like(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols[:max_cols]


def _pick_categorical_cols(df: pd.DataFrame, max_cols: int = 10) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if _is_id_like(c):
            continue
        s = df[c]
        if _is_categorical_like(s):
            nun = int(s.nunique(dropna=True))
            if 1 < nun <= 30:
                cols.append(c)
    return cols[:max_cols]


def _top_abs_corr_pairs(corr: pd.DataFrame, top_k: int = 3) -> List[Tuple[str, str, float]]:
    pairs: List[Tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.isna(v):
                continue
            pairs.append((cols[i], cols[j], float(abs(v))))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _robust_bounds_1d(s: pd.Series, viz_spec: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    """
    Robust bounds using IQR fences (same policy knobs as elsewhere),
    but we always return some bounds (fallback to min/max).
    """
    pol = (viz_spec or {}).get("outlier_policy") or {}
    k = float(pol.get("iqr_k", 1.5))
    min_n = int(pol.get("min_n", 30))

    stats = infer_inlier_bounds(s, k=k, min_n=min_n)
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return 0.0, 1.0, {"ok": False}

    if stats.get("ok", False):
        lo = float(stats["low"])
        hi = float(stats["high"])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return lo, hi, stats

    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return lo, hi, {"ok": False, "min": lo, "max": hi}


def _fit_best_poly_by_aic(x: np.ndarray, y: np.ndarray, max_order: int = 4) -> Tuple[int, np.ndarray]:
    """
    Select polynomial order 1..max_order by AIC on residual sum of squares.
    Guards against too-few points.
    Returns (best_order, coeffs) with coeffs in np.polyfit format.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = int(len(x))
    if n < 3:
        return 1, np.polyfit(x, y, 1)

    best_aic = np.inf
    best_deg = 1
    best_coef = np.polyfit(x, y, 1)

    for deg in range(1, int(max_order) + 1):
        if n < (deg + 2):
            continue
        try:
            coef = np.polyfit(x, y, deg)
            yhat = np.polyval(coef, x)
            resid = y - yhat
            rss = float(np.sum(resid ** 2))
            rss = max(rss, 1e-12)
            k_params = deg + 1
            aic = n * np.log(rss / n) + 2 * k_params
            if aic < best_aic:
                best_aic = aic
                best_deg = deg
                best_coef = coef
        except Exception:
            continue

    return best_deg, best_coef


def _scatter_with_trendline(df: pd.DataFrame, x: str, y: str, viz_spec: Dict[str, Any]) -> Dict[str, Any]:
    # numeric conversion + paired mask
    x_s = pd.to_numeric(df[x], errors="coerce")
    y_s = pd.to_numeric(df[y], errors="coerce")
    paired = x_s.notna() & y_s.notna()
    if int(paired.sum()) < 3:
        return {
            "text": f"Not enough paired numeric values for scatterplot between '{x}' and '{y}'.",
            "figures": [],
            "x": x,
            "y": y,
            "group": None,
            "granularity": None,
        }

    # robust view bounds per-axis
    x_low, x_high, _ = _robust_bounds_1d(x_s[paired], viz_spec)
    y_low, y_high, _ = _robust_bounds_1d(y_s[paired], viz_spec)

    in_view = paired & x_s.between(x_low, x_high) & y_s.between(y_low, y_high)
    out_view = paired & (~in_view)

    x_in = x_s[in_view].astype(float).values
    y_in = y_s[in_view].astype(float).values

    if len(x_in) < 3:
        in_view = paired
        out_view = paired & (~in_view)
        x_in = x_s[in_view].astype(float).values
        y_in = y_s[in_view].astype(float).values

    # Pearson correlation on what is shown (in-view)
    r = float(pd.Series(x_in).corr(pd.Series(y_in)))

    # "below/above" counts like distribution.py, but for 2D:
    # below = any point that falls below the shown rectangle on either axis
    # above = any point that falls above the shown rectangle on either axis
    x_p = x_s[paired]
    y_p = y_s[paired]
    below = int(((x_p < x_low) | (y_p < y_low)).sum())
    above = int(((x_p > x_high) | (y_p > y_high)).sum())

    # Fit best polynomial (1..4) on in-view points
    best_deg, coef = _fit_best_poly_by_aic(x_in, y_in, max_order=4)
    x_line = np.linspace(float(np.min(x_in)), float(np.max(x_in)), 200)
    y_line = np.polyval(coef, x_line)

    # Two-panel layout: main view + excluded points
    fig, (ax, ax_side) = plt.subplots(
        1, 2,
        figsize=(11.8, 5.0),
        gridspec_kw={"width_ratios": [5.0, 1.6]},
        constrained_layout=True,
    )

    # --- Left: main scatter (in-view only)
    ax.scatter(x_in, y_in, alpha=0.70, zorder=2)
    ax.plot(
        x_line,
        y_line,
        linewidth=3.0,
        color="orange",
        alpha=0.95,
        zorder=5,
    )

    # Apply styling without triggering apply_viz's own "outliers hidden" box:
    # - pass df_in (already clipped)
    # - and pin axis limits to the df_in range so apply_viz sees no clipping
    df_in = df.loc[in_view, [x, y]].copy()

    v2 = dict(viz_spec or {})
    ap = dict(v2.get("axis_policy") or {})
    apx = dict(ap.get("x") or {})
    apy = dict(ap.get("y") or {})

    if not df_in.empty:
        xin = pd.to_numeric(df_in[x], errors="coerce").dropna()
        yin = pd.to_numeric(df_in[y], errors="coerce").dropna()
        if not xin.empty:
            apx["limits"] = {"min": float(xin.min()), "max": float(xin.max())}
        if not yin.empty:
            apy["limits"] = {"min": float(yin.min()), "max": float(yin.max())}

    ap["x"] = apx
    ap["y"] = apy
    v2["axis_policy"] = ap

    apply_viz(ax=ax, df=df_in if not df_in.empty else df, viz_spec=v2, x=x, y=y, group=None)

    # Single compact info box (upper-left), exactly like distribution.py wording
    info = f"Pearson cor. = {r:.3f}"
    if below > 0 or above > 0:
        info += "\nOutliers hidden from the graphic:"
        info += f"\n{below} below, {above} above data range"
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.55),
        clip_on=True,
        zorder=10,
    )

    # --- Right: excluded points only (companion plot)
    ax_side.set_title("Removed outliers", fontsize=9, pad=6)
    df_out = df.loc[out_view, [x, y]].copy()
    if df_out.empty:
        ax_side.text(
            0.5, 0.5, "No removed points",
            transform=ax_side.transAxes,
            ha="center", va="center",
            fontsize=9, alpha=0.8
        )
        ax_side.set_xticks([])
        ax_side.set_yticks([])
    else:
        xo = pd.to_numeric(df_out[x], errors="coerce").astype(float).values
        yo = pd.to_numeric(df_out[y], errors="coerce").astype(float).values
        ax_side.scatter(xo, yo, alpha=0.70, zorder=2)

        ax_side.set_xlabel((viz_spec.get("x") or {}).get("label") or x)
        ycfg = (viz_spec.get("y") or {})
        y_label = (ycfg.get("label") or y).strip()
        unit = (viz_spec.get("units") or {}).get(y) or ycfg.get("unit")
        if unit:
            y_label = f"{y_label} ({unit})"
        ax_side.set_ylabel(y_label)

    try:
        ax_side.spines["top"].set_visible(False)
        ax_side.spines["right"].set_visible(False)
        ax_side.grid(False)
    except Exception:
        pass

    text = (
        f"Plotted scatter relationship between '{x}' and '{y}' with a polynomial trendline "
        f"(best order={best_deg}). Pearson cor.={r:.3f}."
    )
    return {"text": text, "figures": [fig], "x": x, "y": y, "group": None, "granularity": None}


# -----------------------------
# KEEP THIS CORRELATION MATRIX AS-IS (unchanged)
# -----------------------------
def _correlation_heatmap(df: pd.DataFrame, num_cols: List[str], viz_spec: Dict[str, Any]) -> Dict[str, Any]:
    sub = df[num_cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(numeric_only=True)

    fig = plt.figure(figsize=(11, 4.8))
    ax = fig.add_subplot(111)

    ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            v = corr.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title((viz_spec or {}).get("title") or "Relationships screening (correlation matrix)")
    apply_viz(ax=ax, df=df, viz_spec=viz_spec, x=None, y=None, group=None)

    top_pairs = _top_abs_corr_pairs(corr, top_k=3)
    if top_pairs:
        parts = [f"{a}~{b} (|r|={v:.2f})" for a, b, v in top_pairs]
        text = "Computed correlation matrix across numeric columns (screening only; no causal claims). Top associations: " + "; ".join(parts) + "."
    else:
        text = "Computed correlation matrix across numeric columns (screening only; no causal claims)."

    return {"text": text, "figures": [fig], "x": None, "y": None, "group": None, "granularity": None}


def _crosstab_heatmap(df: pd.DataFrame, a: str, b: str, viz_spec: Dict[str, Any]) -> Dict[str, Any]:
    s1 = df[a].astype("string").str.strip()
    s2 = df[b].astype("string").str.strip()

    ct = pd.crosstab(s1, s2, dropna=False)
    if ct.shape[0] > 20:
        top_rows = ct.sum(axis=1).sort_values(ascending=False).index[:20]
        ct = ct.loc[top_rows]
    if ct.shape[1] > 20:
        top_cols = ct.sum(axis=0).sort_values(ascending=False).index[:20]
        ct = ct[top_cols]

    ct_norm = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig = plt.figure(figsize=(11, 4.8))
    ax = fig.add_subplot(111)

    ax.imshow(ct_norm.values, aspect="auto")
    ax.set_xticks(range(ct_norm.shape[1]))
    ax.set_yticks(range(ct_norm.shape[0]))
    ax.set_xticklabels([str(x) for x in ct_norm.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(x) for x in ct_norm.index])

    ax.set_title((viz_spec or {}).get("title") or f"Relationships screening: {a} vs {b} (row-normalized)")
    apply_viz(ax=ax, df=df, viz_spec=viz_spec, x=None, y=None, group=None)

    text = f"Computed crosstab screening for '{a}' vs '{b}' (row-normalized shares; screening only)."
    return {"text": text, "figures": [fig], "x": a, "y": b, "group": None, "granularity": None}


def run_relationships(*, df: pd.DataFrame, viz_spec: Dict[str, Any], aggregate_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data to analyze (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    mode = (viz_spec or {}).get("relationships_mode")
    if isinstance(mode, str):
        mode = mode.strip().lower()
    else:
        mode = None

    num_cols = _pick_numeric_cols(df, max_cols=10)
    cat_cols = _pick_categorical_cols(df, max_cols=10)

    if mode == "pair":
        if len(num_cols) >= 2:
            return _scatter_with_trendline(df, num_cols[0], num_cols[1], viz_spec)

    if mode == "matrix":
        if len(num_cols) >= 3:
            return _correlation_heatmap(df, num_cols, viz_spec)

    # auto
    if len(num_cols) == 2:
        return _scatter_with_trendline(df, num_cols[0], num_cols[1], viz_spec)
    if len(num_cols) >= 3:
        return _correlation_heatmap(df, num_cols, viz_spec)

    if len(cat_cols) >= 2:
        return _crosstab_heatmap(df, cat_cols[0], cat_cols[1], viz_spec)

    return {"text": "Not enough suitable columns for relationships screening (need >=2 numeric or >=2 categorical columns).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}