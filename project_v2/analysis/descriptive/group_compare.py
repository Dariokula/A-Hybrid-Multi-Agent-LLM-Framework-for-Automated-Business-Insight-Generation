# analysis/descriptive/group_compare.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

from analysis.viz_apply import apply_viz, infer_inlier_bounds


MAX_CATEGORIES = 20
MAX_LABEL_CHARS = 20


def _is_categorical_like(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    )


def _truncate_label(x: Any, max_chars: int = MAX_LABEL_CHARS) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + " [...]"


def _pick_y(df: pd.DataFrame, resolved_y: Optional[str]) -> Optional[str]:
    if isinstance(resolved_y, str) and resolved_y in df.columns and pd.api.types.is_numeric_dtype(df[resolved_y]):
        return resolved_y
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _pick_x_for_compare(
    df: pd.DataFrame,
    resolved_x: Optional[str],
    resolved_group: Optional[str],
    y: Optional[str],
) -> Optional[str]:
    candidates: List[str] = []
    for c in df.columns:
        if c in {y}:
            continue
        if c == "time_bucket":
            continue
        if _is_categorical_like(df[c]):
            nun = int(df[c].nunique(dropna=True))
            if 1 < nun <= 30:
                candidates.append(c)

    if isinstance(resolved_group, str) and resolved_group in candidates:
        return resolved_group
    if isinstance(resolved_x, str) and resolved_x in candidates:
        return resolved_x
    return candidates[0] if candidates else None


def _top_k_groups_by_mean(df: pd.DataFrame, *, x: str, y: str, k: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      - plot_df: [x, y] mean per group, sorted desc, limited to top-k
      - kept_groups: Index/Series of kept group labels
    """
    tmp = df.groupby(x, dropna=False)[y].mean().sort_values(ascending=False)
    keep = tmp.index[:k]
    plot_df = tmp.loc[keep].reset_index()
    plot_df = plot_df.rename(columns={y: y})
    plot_df = plot_df.sort_values(y, ascending=False)
    return plot_df, pd.Index(keep)


def run_group_compare(*, df: pd.DataFrame, viz_spec: Dict[str, Any], aggregate_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data to plot (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    resolved = (viz_spec or {}).get("resolved") or {}
    y = _pick_y(df, resolved.get("y"))
    if not y or y not in df.columns:
        return {"text": "No numeric metric available for group comparison.", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    x = _pick_x_for_compare(df, resolved.get("x"), resolved.get("group"), y)
    if not x or x not in df.columns:
        return {"text": "No suitable categorical column found for group comparison.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    df2 = df.copy()
    if pd.api.types.is_object_dtype(df2[x]) or pd.api.types.is_string_dtype(df2[x]):
        df2[x] = df2[x].astype("string").str.strip()

    # Compute mean per group on full data, but only show top MAX_CATEGORIES groups.
    plot_df, kept = _top_k_groups_by_mean(df2, x=x, y=y, k=MAX_CATEGORIES)

    # Ensure stable string labels for plotting and for text summary
    plot_df[x] = plot_df[x].astype("string")

    # Plot using numeric positions so we can force "label every bar"
    n = int(len(plot_df))
    positions = np.arange(n, dtype=float)

    labels_full = plot_df[x].tolist()
    labels_short = [_truncate_label(v, MAX_LABEL_CHARS) for v in labels_full]

    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(111)

    ax.bar(positions, plot_df[y].values.astype(float))

    # Apply global viz styling/labels, but do NOT let it decide categorical ticks.
    # We pass x=None so apply_viz won't touch x tick locator/formatter.
    apply_viz(ax=ax, df=plot_df, viz_spec=viz_spec, x=None, y=y, group=None)

    # Set x label ourselves (since x=None above)
    x_label = ((viz_spec.get("x") or {}).get("label") or x).strip()
    ax.set_xlabel(x_label)

    # Force every bar to have a tick+label (prevents MaxNLocator from dropping them)
    ax.xaxis.set_major_locator(FixedLocator(positions))
    ax.xaxis.set_major_formatter(FixedFormatter(labels_short))

    for t in ax.get_xticklabels():
        t.set_rotation(45)
        t.set_ha("right")

    # If many categories, reduce font a bit but keep all labels
    if n >= 12:
        for t in ax.get_xticklabels():
            t.set_fontsize(8)

    # Summary text
    if n >= 2:
        best = plot_df.iloc[0]
        worst = plot_df.iloc[-1]
        gap = float(best[y]) - float(worst[y])
        nun_total = int(df2[x].nunique(dropna=True))
        shown = int(n)
        truncated_note = f" (top {shown} of {nun_total} groups shown)" if nun_total > shown else f" ({shown} groups shown)"
        text = (
            f"Compared '{y}' across '{x}' using mean-per-group values{truncated_note}. "
            f"Highest: {best[x]}={float(best[y]):.4g}; Lowest: {worst[x]}={float(worst[y]):.4g}; Gap≈{gap:.4g}."
        )
    else:
        nun_total = int(df2[x].nunique(dropna=True))
        text = f"Compared '{y}' across '{x}' (only {n} group shown; total groups in data: {nun_total})."

    # Optional: small hint about raw-data outliers (keep as-is, but ensure it doesn't conflict with plot title)
    try:
        b = infer_inlier_bounds(
            df2[y],
            k=float((viz_spec.get("outlier_policy") or {}).get("iqr_k", 1.5)),
            min_n=int((viz_spec.get("outlier_policy") or {}).get("min_n", 30)),
        )
        if b.get("ok", False) and int(b.get("outlier_count", 0)) > 0:
            ax.text(
                0.02,
                0.98,
                f"Raw outliers (IQR): {int(b['outlier_count'])} ({float(b['outlier_frac']):.1%})\n"
                f"Raw range: [{float(b['min']):.4g}, {float(b['max']):.4g}]",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", alpha=0.55),
            )
    except Exception:
        pass

    plan = aggregate_ctx.get("plan") if isinstance(aggregate_ctx, dict) else None
    if not isinstance(plan, dict):
        plan = aggregate_ctx if isinstance(aggregate_ctx, dict) else {}
    gran = plan.get("time_granularity")

    return {"text": text, "figures": [fig], "x": x, "y": y, "group": x, "granularity": gran}