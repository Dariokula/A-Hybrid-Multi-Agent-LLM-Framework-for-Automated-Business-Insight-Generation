# analysis/diagnostic/anomaly_explanation.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _is_id_like(col: str) -> bool:
    cl = col.lower()
    return any(k in cl for k in ["id", "uuid", "guid", "key", "nr", "no.", "number", "pos"])


def _is_categorical_like(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    )


def _pick_numeric_y(df: pd.DataFrame, resolved_y: Optional[str]) -> Optional[str]:
    if isinstance(resolved_y, str) and resolved_y in df.columns and pd.api.types.is_numeric_dtype(df[resolved_y]):
        return resolved_y
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(c):
            return c
    return None


def _iqr_bounds(s: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return (np.nan, np.nan)
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return (q1, q3)
    return (q1 - k * iqr, q3 + k * iqr)


def _infer_mode(prompt: Optional[str], viz_spec: Dict[str, Any]) -> str:
    """
    Returns: 'high' | 'low' | 'both'
    Priority: viz_spec['anomaly_mode'] if present (set in analyze.py), else prompt heuristic.
    """
    m = (viz_spec or {}).get("anomaly_mode")
    if isinstance(m, str) and m.strip().lower() in {"high", "low", "both"}:
        return m.strip().lower()

    p = (prompt or "").lower()

    high_markers = [
        "late", "delay", "delays", "behind", "overdue", "exceed", "exceeds", "largest", "highest", "worst",
        "extreme late", "most extreme late", "most late",
        "spät", "verspät", "verzug", "zu spät", "am spätesten",
        "high outlier", "upper tail", "top outlier",
    ]
    low_markers = [
        "early", "ahead", "negative", "smallest", "lowest", "best", "shortest", "fastest",
        "früh", "zu früh", "am frühesten",
        "low outlier", "lower tail", "bottom outlier",
    ]

    has_high = any(m in p for m in high_markers)
    has_low = any(m in p for m in low_markers)

    if has_high and not has_low:
        return "high"
    if has_low and not has_high:
        return "low"
    return "both"


def _top_categorical_diffs(out_df: pd.DataFrame, base_df: pd.DataFrame, *, y: str, max_features: int = 8) -> pd.DataFrame:
    """
    For each categorical column, pick the single value with the largest share difference:
      Δpp = (share_outliers - share_typical) * 100
    """
    rows: List[Dict[str, Any]] = []
    for c in out_df.columns:
        if c == y or _is_id_like(c):
            continue
        if _is_categorical_like(out_df[c]):
            nun = int(pd.Series(pd.concat([out_df[c], base_df[c]], axis=0)).nunique(dropna=True))
            if not (2 <= nun <= 30):
                continue

            a = out_df[c].astype("string").str.strip()
            b = base_df[c].astype("string").str.strip()
            a_share = a.value_counts(dropna=False) / max(1, len(a))
            b_share = b.value_counts(dropna=False) / max(1, len(b))

            all_vals = set(a_share.index.tolist()) | set(b_share.index.tolist())
            best_val = None
            best_diff = 0.0
            best_a = 0.0
            best_b = 0.0
            for v in all_vals:
                da = float(a_share.get(v, 0.0))
                db = float(b_share.get(v, 0.0))
                diff = da - db
                if abs(diff) > abs(best_diff):
                    best_diff = diff
                    best_val = v
                    best_a = da
                    best_b = db

            if best_val is None:
                continue

            rows.append(
                {
                    "feature": c,
                    "value": str(best_val),
                    "diff_frac": float(best_diff),
                    "diff_pp": float(best_diff) * 100.0,
                    "abs_pp": float(abs(best_diff)) * 100.0,
                    "share_out": float(best_a) * 100.0,
                    "share_base": float(best_b) * 100.0,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("abs_pp", ascending=False).head(max_features).copy()
    df["label"] = (
        df["feature"].astype(str)
        + "="
        + df["value"].astype(str)
        + " (Δ="
        + df["diff_pp"].map(lambda v: f"{v:+.1f}pp")
        + ")"
    )
    return df


def _top_numeric_outlier_association(full_df: pd.DataFrame, out_mask: pd.Series, *, y: str, max_features: int = 8) -> pd.DataFrame:
    """
    For numeric columns: compute Pearson correlation r between the feature and the outlier-flag (0/1).
    This is easy to explain:
      r > 0  => higher feature values tend to appear in outliers
      r < 0  => lower feature values tend to appear in outliers
    """
    rows: List[Dict[str, Any]] = []
    flag = out_mask.astype(int)

    for c in full_df.columns:
        if c == y or _is_id_like(c):
            continue
        if pd.api.types.is_numeric_dtype(full_df[c]):
            x = pd.to_numeric(full_df[c], errors="coerce")
            m = x.notna() & flag.notna()
            if int(m.sum()) < 30:
                continue
            # Need both classes present for meaningful correlation
            if flag[m].nunique() < 2:
                continue

            r = float(pd.concat([x[m], flag[m]], axis=1).corr().iloc[0, 1])
            if not np.isfinite(r):
                continue

            # Add simple means (for human readability)
            mean_out = float(x[m & (flag == 1)].mean())
            mean_base = float(x[m & (flag == 0)].mean())

            rows.append(
                {
                    "feature": c,
                    "r": r,
                    "abs_r": abs(r),
                    "mean_out": mean_out,
                    "mean_base": mean_base,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("abs_r", ascending=False).head(max_features).copy()
    df["label"] = df["feature"].astype(str) + " (r=" + df["r"].map(lambda v: f"{v:+.2f}") + ")"
    return df


def _tufte_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.18)
    ax.grid(False, axis="x")


def _format_box_counts(out_low: int, out_high: int) -> str:
    return (
        "Outliers hidden from the graphic:\n"
        f"{int(out_low)} below, {int(out_high)} above data range"
    )


def _choose_bins(x: np.ndarray, max_bins: int = 40) -> int:
    n = int(len(x))
    if n <= 1:
        return 10
    b = int(np.sqrt(n) * 1.2)
    return int(max(10, min(max_bins, b)))


def run_anomaly_explanation(
    *,
    df: pd.DataFrame,
    viz_spec: Dict[str, Any],
    aggregate_ctx: Dict[str, Any],
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Baseline-safe outlier explanation (no dataset filtering).

    Supports modes derived from prompt:
      - high  -> focus on upper-tail outliers
      - low   -> focus on lower-tail outliers
      - both  -> both tails

    Output:
      Figure 1: Inlier view (with one "outliers hidden" box) + Outliers-only view
      Figure 2: Easy-to-explain driver view:
                - Categorical: share difference in percentage points (Δpp)
                - Numeric: correlation r with "being an outlier" (flag 0/1)
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    resolved = (viz_spec or {}).get("resolved") or {}
    y = _pick_numeric_y(df, resolved.get("y"))
    if not y or y not in df.columns:
        return {"text": "No numeric KPI column found for anomaly explanation.", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    mode = _infer_mode(prompt, viz_spec)

    df2 = df.copy()
    y_s = pd.to_numeric(df2[y], errors="coerce")
    df2 = df2.loc[y_s.notna()].copy()
    if df2.empty:
        return {"text": f"No numeric values available in '{y}' for anomaly explanation.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    x = pd.to_numeric(df2[y], errors="coerce").astype(float)

    # Define baseline-safe outlier bounds
    low, high = _iqr_bounds(x, k=float((viz_spec.get("outlier_policy") or {}).get("iqr_k", 1.5)))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(x.quantile(0.01))
        high = float(x.quantile(0.99))

    if mode == "high":
        out_mask = x > high
    elif mode == "low":
        out_mask = x < low
    else:
        out_mask = (x < low) | (x > high)

    out_n = int(out_mask.sum())
    n = int(len(df2))

    # If too few outliers, widen slightly using 5% tails in the requested direction (still baseline-safe)
    if out_n < 8 and n >= 40:
        q05 = float(x.quantile(0.05))
        q95 = float(x.quantile(0.95))
        if mode == "high":
            high = q95
            out_mask = x > high
        elif mode == "low":
            low = q05
            out_mask = x < low
        else:
            low, high = q05, q95
            out_mask = (x < low) | (x > high)
        out_n = int(out_mask.sum())

    out_df = df2.loc[out_mask].copy()
    base_df = df2.loc[~out_mask].copy()

    if out_n == 0 or base_df.empty:
        return {
            "text": f"No outliers found for '{y}' under the current rule (mode={mode}).",
            "figures": [],
            "x": None,
            "y": y,
            "group": None,
            "granularity": None,
        }

    # ---------------------------
    # Figure 1: inlier view + outliers-only
    # ---------------------------
    inlier_mask = (~((x < low) | (x > high)))  # view-range based on two-sided fences
    inliers = x.loc[inlier_mask].values

    # If view inliers are too few (degenerate), fall back to central 98% as view
    if len(inliers) < 20 and n >= 40:
        vlow = float(x.quantile(0.01))
        vhigh = float(x.quantile(0.99))
        inlier_mask = (x >= vlow) & (x <= vhigh)
        inliers = x.loc[inlier_mask].values

    out_low = int((x < float(np.min(inliers)) if len(inliers) else x < low).sum())
    out_high = int((x > float(np.max(inliers)) if len(inliers) else x > high).sum())

    outliers_selected = x.loc[out_mask].values  # mode-specific outliers-only plot

    fig1, (axL, axR) = plt.subplots(
        1, 2,
        figsize=(12.4, 4.8),
        gridspec_kw={"width_ratios": [3.0, 2.2]},
        constrained_layout=True,
    )

    # Left: inlier histogram
    binsL = _choose_bins(inliers)
    axL.hist(inliers, bins=binsL, rwidth=0.98, alpha=0.85)
    _tufte_axes(axL)
    axL.set_title(f"Typical range (baseline view) — KPI: {y}", fontsize=10, pad=10)
    axL.set_xlabel(y)
    axL.set_ylabel("Count")

    # mean/median markers + labels
    med = float(np.median(inliers)) if len(inliers) else float(x.median())
    mean = float(np.mean(inliers)) if len(inliers) else float(x.mean())
    axL.axvline(med, linestyle="--", linewidth=1.3, alpha=0.9)
    axL.axvline(mean, linestyle="--", linewidth=1.3, alpha=0.9)
    ytop = axL.get_ylim()[1]
    axL.text(med, ytop * 0.92, "Med", ha="center", va="top", fontsize=8)
    axL.text(mean, ytop * 0.82, "Mean", ha="center", va="top", fontsize=8)

    # single info box (upper left)
    axL.text(
        0.02, 0.98,
        _format_box_counts(out_low, out_high),
        transform=axL.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.28", alpha=0.65),
        clip_on=True,
    )

    # Right: outliers-only histogram
    title_map = {"high": "High outliers", "low": "Low outliers", "both": "Outliers"}
    axR.set_title(title_map.get(mode, "Outliers"), fontsize=10, pad=10)

    if outliers_selected.size > 0:
        binsR = _choose_bins(outliers_selected, max_bins=30)
        axR.hist(outliers_selected, bins=binsR, rwidth=0.98, alpha=0.85)
        _tufte_axes(axR)
        axR.set_xlabel(y)
        axR.set_ylabel("Count")

        med_o = float(np.median(outliers_selected))
        mean_o = float(np.mean(outliers_selected))
        axR.axvline(med_o, linestyle="--", linewidth=1.3, alpha=0.9)
        axR.axvline(mean_o, linestyle="--", linewidth=1.3, alpha=0.9)
        ytop2 = axR.get_ylim()[1]
        axR.text(med_o, ytop2 * 0.92, "Med", ha="center", va="top", fontsize=8)
        axR.text(mean_o, ytop2 * 0.82, "Mean", ha="center", va="top", fontsize=8)
    else:
        axR.text(0.5, 0.5, "No outliers in this mode", ha="center", va="center")
        axR.set_xticks([])
        axR.set_yticks([])

    # ---------------------------
    # Figure 2: easy-to-explain drivers
    # ---------------------------
    cat_df = _top_categorical_diffs(out_df, base_df, y=y, max_features=8)
    num_df = _top_numeric_outlier_association(df2, out_mask, y=y, max_features=8)

    fig2, (axC, axN) = plt.subplots(
        1, 2,
        figsize=(12.6, 4.6),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )

    # Categorical panel (Δpp)
    _tufte_axes(axC)
    axC.set_title("Categorical drivers (share difference)", fontsize=10, pad=10)
    if cat_df.empty:
        axC.text(0.5, 0.5, "No clear categorical driver found.", ha="center", va="center", fontsize=10)
        axC.set_xticks([])
        axC.set_yticks([])
    else:
        ddf = cat_df.sort_values("abs_pp", ascending=True).tail(10)
        axC.barh(ddf["label"].tolist(), ddf["diff_pp"].astype(float).values)
        axC.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
        axC.set_xlabel("Δ percentage points (outliers − typical)")

    # Numeric panel (correlation with outlier flag)
    _tufte_axes(axN)
    axN.set_title("Numeric drivers (association with being an outlier)", fontsize=10, pad=10)
    if num_df.empty:
        axN.text(0.5, 0.5, "No clear numeric driver found.", ha="center", va="center", fontsize=10)
        axN.set_xticks([])
        axN.set_yticks([])
    else:
        ddf = num_df.sort_values("abs_r", ascending=True).tail(10)
        axN.barh(ddf["label"].tolist(), ddf["r"].astype(float).values)
        axN.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
        axN.set_xlim(-1.0, 1.0)
        axN.set_xlabel("Correlation r with outlier flag (−1…+1)")

        # Small in-plot note (not covering data)
        axN.text(
            0.02, 0.02,
            "r>0: higher values in outliers\nr<0: lower values in outliers",
            transform=axN.transAxes,
            ha="left", va="bottom",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.55),
            clip_on=True,
        )

    # ---------------------------
    # Text summary (plain language)
    # ---------------------------
    out_vals = pd.to_numeric(out_df[y], errors="coerce").dropna()
    base_vals = pd.to_numeric(base_df[y], errors="coerce").dropna()
    mean_out = float(out_vals.mean()) if not out_vals.empty else float("nan")
    mean_base = float(base_vals.mean()) if not base_vals.empty else float("nan")

    mode_label = {"high": "high outliers", "low": "low outliers", "both": "outliers"}.get(mode, "outliers")

    parts = [
        f"Explained {mode_label} for '{y}' without filtering the dataset (baseline preserved).",
        f"Found {out_n} outliers out of {n} records.",
        f"Average KPI: outliers≈{mean_out:.4g} vs typical≈{mean_base:.4g}.",
        "Figure 1 shows the typical range and an outliers-only view.",
        "Figure 2 shows two simple driver views: categories in percentage points (Δpp) and numeric columns via correlation with the outlier flag (r).",
    ]

    # Add top drivers in readable form
    if not cat_df.empty:
        r0 = cat_df.iloc[0]
        parts.append(f"Top categorical driver: {r0['feature']}={r0['value']} (Δ≈{float(r0['diff_pp']):+.1f}pp).")
    if not num_df.empty:
        r1 = num_df.iloc[0]
        parts.append(f"Top numeric driver: {r1['feature']} (r≈{float(r1['r']):+.2f}).")

    text = " ".join(parts)

    return {"text": text, "figures": [fig1, fig2], "x": None, "y": y, "group": None, "granularity": None}