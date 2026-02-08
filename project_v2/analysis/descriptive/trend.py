# analysis/descriptive/trend.py
from __future__ import annotations
from typing import Any, Dict

import pandas as pd
import matplotlib.pyplot as plt

from analysis.viz_apply import apply_viz


def run_trend(*, df: pd.DataFrame, viz_spec: Dict[str, Any], aggregate_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data to plot (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    resolved = (viz_spec or {}).get("resolved") or {}
    x = resolved.get("x")
    y = resolved.get("y")
    group = resolved.get("group")

    if not x or x not in df.columns:
        return {"text": "No x column available for trend.", "figures": [], "x": x, "y": None, "group": group, "granularity": None}
    if not y or y not in df.columns:
        return {"text": "No y column available for trend.", "figures": [], "x": x, "y": y, "group": group, "granularity": None}

    df2 = df.copy()
    df2[x] = pd.to_datetime(df2[x], errors="coerce")
    df2 = df2.dropna(subset=[x])
    if df2.empty:
        return {"text": "No valid timestamps to plot.", "figures": [], "x": x, "y": y, "group": group, "granularity": None}

    # Keep full data; view robustness is handled via robust axis limits + clipping annotation.
    df_plot = df2.sort_values(x)

    def _scaled_sizes(n: pd.Series, s_min: float = 25.0, s_max: float = 180.0) -> pd.Series:
        """Map support counts to scatter marker sizes (stable across small ranges)."""
        nn = pd.to_numeric(n, errors="coerce").fillna(0.0)
        if len(nn) == 0:
            return pd.Series([], dtype=float)
        lo = float(nn.min())
        hi = float(nn.max())
        if hi <= lo:
            return pd.Series([0.5 * (s_min + s_max)] * len(nn), index=nn.index)
        return s_min + (nn - lo) * (s_max - s_min) / (hi - lo)

    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(111)

    has_support = "n_records" in df_plot.columns

    if group and group in df_plot.columns and df_plot[group].nunique(dropna=True) > 1:
        if pd.api.types.is_string_dtype(df_plot[group]) or pd.api.types.is_object_dtype(df_plot[group]):
            df_plot[group] = df_plot[group].astype("string").str.strip()
        for g, sub in df_plot.groupby(group, dropna=False):
            # line for trend shape
            ax.plot(sub[x], sub[y], linewidth=1.6, label=str(g))
            # size-encoded points for bucket support
            if has_support:
                ax.scatter(sub[x], sub[y], s=_scaled_sizes(sub["n_records"]), alpha=0.85)
            else:
                ax.scatter(sub[x], sub[y], s=45, alpha=0.85)
    else:
        ax.plot(df_plot[x], df_plot[y], linewidth=1.6)
        if has_support:
            ax.scatter(df_plot[x], df_plot[y], s=_scaled_sizes(df_plot["n_records"]), alpha=0.85)
        else:
            ax.scatter(df_plot[x], df_plot[y], s=45, alpha=0.85)

    if has_support:
        ax.text(
            0.99, 0.02,
            "Data point size implies amount of aggregated data",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8,
            alpha=0.8,
        )

    apply_viz(ax=ax, df=df_plot, viz_spec=viz_spec, x=x, y=y, group=group)

    plan = aggregate_ctx.get("plan") if isinstance(aggregate_ctx, dict) else None
    if not isinstance(plan, dict):
        plan = aggregate_ctx if isinstance(aggregate_ctx, dict) else {}
    gran = plan.get("time_granularity")

    text = f"Plotted trend for '{y}' over '{x}' ({len(df_plot)} rows)."
    return {"text": text, "figures": [fig], "x": x, "y": y, "group": group, "granularity": gran}