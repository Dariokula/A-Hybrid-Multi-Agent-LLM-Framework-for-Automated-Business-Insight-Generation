# analysis/descriptive/stats_summary.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.viz_apply import apply_viz, infer_inlier_bounds


# -----------------------------------------------------------------------------
# Distribution.py helpers (copied to keep the left two plots identical)
# -----------------------------------------------------------------------------
def _is_categorical_like(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    )


def _pick_group(df: pd.DataFrame, resolved_group: Optional[str], y: Optional[str]) -> Optional[str]:
    if isinstance(resolved_group, str) and resolved_group in df.columns and resolved_group != y:
        return resolved_group
    for c in df.columns:
        if c == y:
            continue
        s = df[c]
        if _is_categorical_like(s):
            nun = int(s.nunique(dropna=True))
            if 1 < nun <= 20:
                return c
    return None


def _quiet_axes(ax: plt.Axes) -> None:
    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass


def _integer_like(values: np.ndarray, *, atol: float = 1e-9) -> bool:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return False
    return bool(np.all(np.isclose(v, np.round(v), atol=atol)))


def _relax_singleton_inlier_bounds(
    s_all: pd.Series,
    low: float,
    high: float,
    s_in: pd.Series,
) -> Tuple[float, float, pd.Series]:
    """
    Edge-case fix (same as distribution.py):
    If outlier trimming collapses the inlier view to a single unique value while the
    full data has >1 unique value, relax bounds to show at least two classes when possible.

    Strategy:
      1) For integer-like data, round bounds outward: low=floor(low), high=ceil(high)
      2) If still singleton, expand to include the nearest distinct value adjacent to the singleton.
    """
    try:
        nun_all = int(s_all.nunique(dropna=True))
        nun_in = int(s_in.nunique(dropna=True))
    except Exception:
        return low, high, s_in

    if nun_all <= 1 or nun_in > 1:
        return low, high, s_in

    vals_all = s_all.to_numpy(dtype=float)
    vals_all = vals_all[np.isfinite(vals_all)]
    if vals_all.size == 0:
        return low, high, s_in

    # 1) Integer-like: round outward to avoid slicing off integer bins.
    if _integer_like(vals_all):
        low2 = float(np.floor(low))
        high2 = float(np.ceil(high))
        s_in2 = s_all[(s_all >= low2) & (s_all <= high2)]
        if int(s_in2.nunique(dropna=True)) > 1:
            return low2, high2, s_in2
        low, high, s_in = low2, high2, s_in2

    # 2) Still singleton: include nearest distinct value.
    uniq = np.unique(vals_all)
    if uniq.size <= 1:
        return low, high, s_in

    in_vals = s_in.to_numpy(dtype=float)
    in_vals = in_vals[np.isfinite(in_vals)]
    if in_vals.size == 0:
        return low, high, s_in

    anchor = float(np.unique(in_vals)[0])

    idx = int(np.searchsorted(uniq, anchor))
    if idx < uniq.size and np.isclose(uniq[idx], anchor):
        anchor_idx = idx
    elif idx > 0 and np.isclose(uniq[idx - 1], anchor):
        anchor_idx = idx - 1
    else:
        anchor_idx = int(np.argmin(np.abs(uniq - anchor)))
        anchor = float(uniq[anchor_idx])

    below_val = float(uniq[anchor_idx - 1]) if anchor_idx - 1 >= 0 else None
    above_val = float(uniq[anchor_idx + 1]) if anchor_idx + 1 < uniq.size else None

    below_out = int((s_all < low).sum())
    above_out = int((s_all > high).sum())

    low_new, high_new = low, high

    if below_out > 0 and below_val is not None:
        low_new = min(low_new, below_val)
    if above_out > 0 and above_val is not None:
        high_new = max(high_new, above_val)

    if np.isclose(low_new, low) and np.isclose(high_new, high):
        cand = []
        if below_val is not None:
            cand.append(("below", abs(anchor - below_val), below_val))
        if above_val is not None:
            cand.append(("above", abs(above_val - anchor), above_val))
        cand.sort(key=lambda x: x[1])
        if cand:
            side, _, v = cand[0]
            if side == "below":
                low_new = min(low_new, v)
            else:
                high_new = max(high_new, v)

    if not np.isfinite(low_new) or not np.isfinite(high_new) or high_new < low_new:
        return low, high, s_in

    s_in3 = s_all[(s_all >= low_new) & (s_all <= high_new)]
    if int(s_in3.nunique(dropna=True)) > 1 or len(s_in3) > len(s_in):
        return float(low_new), float(high_new), s_in3

    return low, high, s_in


def _inlier_view_range(
    s_all: pd.Series,
    bounds: Dict[str, Any],
) -> Tuple[float, float, pd.Series, pd.Series]:
    """Return (low_view, high_view, s_inliers, s_outliers) based on robust bounds."""
    if bounds.get("ok", False):
        low = float(bounds["low"])
        high = float(bounds["high"])
    else:
        low = float(s_all.min())
        high = float(s_all.max())

    s_in = s_all[(s_all >= low) & (s_all <= high)]
    s_out = s_all[(s_all < low) | (s_all > high)]

    if s_in.empty:
        low = float(s_all.min())
        high = float(s_all.max())
        s_in = s_all
        s_out = s_all.iloc[0:0]
        return low, high, s_in, s_out

    # ✅ New: relax singleton inlier view (same behavior as distribution.py)
    low2, high2, s_in2 = _relax_singleton_inlier_bounds(s_all, low, high, s_in)
    if (not np.isclose(low2, low)) or (not np.isclose(high2, high)) or (len(s_in2) != len(s_in)):
        low, high, s_in = low2, high2, s_in2
        s_out = s_all[(s_all < low) | (s_all > high)]

    return low, high, s_in, s_out


def _hist_edges_for_view(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Histogram bin edges with integer alignment + padding.

    For integer-like data:
      left = floor(low)-1
      right = ceil(high)+1
      edges are half-integers so bars center on integer values.
    """
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.array([0.0, 1.0])

    if _integer_like(vals):
        lo = float(np.floor(low)) - 1.0
        hi = float(np.ceil(high)) + 1.0
        return np.arange(lo - 0.5, hi + 0.5 + 1.0, 1.0)

    lo = float(low)
    hi = float(high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0

    n = int(vals.size)
    n_bins = int(min(50, max(12, round(np.sqrt(n) * 1.2))))
    return np.linspace(lo, hi, n_bins + 1)


def _annotate_outliers_hidden(ax: plt.Axes, below: int, above: int) -> None:
    """Single box (top corner), placed to avoid clashing with center-line labels when possible."""
    below = int(max(0, below))
    above = int(max(0, above))
    if below <= 0 and above <= 0:
        return

    txt = "Outliers hidden from the graphic:\n"
    txt += f"{below} below, {above} above data range"

    place_right = True
    try:
        xlim = ax.get_xlim()
        if np.isfinite(xlim[0]) and np.isfinite(xlim[1]) and xlim[1] > xlim[0]:
            mid = (float(xlim[0]) + float(xlim[1])) / 2.0
            xs = []
            for ln in getattr(ax, "lines", []):
                try:
                    xd = np.asarray(ln.get_xdata(), dtype=float)
                    if xd.size > 0 and np.isfinite(xd[0]) and np.isfinite(xd[-1]) and np.isclose(xd[0], xd[-1]):
                        xs.append(float(xd[0]))
                except Exception:
                    continue
            if xs and np.nanmean(xs) > mid:
                place_right = False
    except Exception:
        pass

    if place_right:
        x, ha = 0.98, "right"
    else:
        x, ha = 0.02, "left"

    ax.text(
        x,
        0.98,
        txt,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.30", alpha=0.60),
        clip_on=True,
    )


def _draw_center_lines_with_labels(ax: plt.Axes, s_in: pd.Series) -> None:
    vals = s_in.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    med = float(np.nanmedian(vals))
    avg = float(np.nanmean(vals))

    ax.axvline(med, linewidth=1.2, alpha=0.60)
    ax.axvline(avg, linewidth=1.1, alpha=0.45, linestyle="--")

    ax.text(med, 0.98, "Med", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, alpha=0.85)
    ax.text(avg, 0.98, "Avg", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, alpha=0.75)


def _make_side_full_range_box(ax_side: plt.Axes, values: np.ndarray) -> None:
    """Middle panel: boxplot incl. fliers. Styled to avoid thick/colored artifacts."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        ax_side.text(0.5, 0.5, "No data", transform=ax_side.transAxes, ha="center", va="center", fontsize=9, alpha=0.8)
        ax_side.set_xticks([])
        ax_side.set_yticks([])
        _quiet_axes(ax_side)
        return

    ax_side.boxplot(
        [vals],
        showfliers=True,
        vert=False,
        widths=0.55,
        patch_artist=False,
        medianprops=dict(linewidth=1.1, alpha=0.9),
        whiskerprops=dict(linewidth=1.0, alpha=0.85),
        capprops=dict(linewidth=1.0, alpha=0.85),
        boxprops=dict(linewidth=1.0, alpha=0.85),
        flierprops=dict(markersize=3, alpha=0.65),
    )
    ax_side.set_yticks([])
    ax_side.set_title("Removed outliers", fontsize=9, pad=6)
    ax_side.grid(False)
    _quiet_axes(ax_side)


# -----------------------------------------------------------------------------
# Stats helpers (kept from your previous stats_summary approach)
# -----------------------------------------------------------------------------
def _is_id_like(col: str) -> bool:
    cl = str(col).lower()
    return any(k in cl for k in ["id", "uuid", "guid", "key"])


def _format_num(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        xf = float(x)
        if abs(xf) >= 1000:
            return f"{xf:,.2f}"
        if abs(xf) >= 10:
            return f"{xf:.2f}"
        return f"{xf:.4f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def _stats(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    n_total = int(len(s))
    s_valid = s.dropna()
    n = int(len(s_valid))
    miss = 0.0 if n_total == 0 else float(1.0 - (n / max(1, n_total)))

    if n == 0:
        return {
            "n_total": n_total,
            "n": 0,
            "missing_rate": 1.0 if n_total > 0 else 0.0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "n_total": n_total,
        "n": n,
        "missing_rate": miss,
        "mean": float(s_valid.mean()),
        "median": float(s_valid.median()),
        "std": float(s_valid.std(ddof=1)) if n > 1 else 0.0,
        "min": float(s_valid.min()),
        "p05": float(s_valid.quantile(0.05)),
        "p25": float(s_valid.quantile(0.25)),
        "p75": float(s_valid.quantile(0.75)),
        "p95": float(s_valid.quantile(0.95)),
        "max": float(s_valid.max()),
    }


def _unit_for(viz_spec: Dict[str, Any], col: str) -> Optional[str]:
    if not isinstance(viz_spec, dict):
        return None
    units = viz_spec.get("units")
    if isinstance(units, dict) and col in units and units[col]:
        return str(units[col])
    ycfg = viz_spec.get("y") if isinstance(viz_spec.get("y"), dict) else {}
    u = ycfg.get("unit")
    return str(u) if u else None


def _make_stats_card(ax: plt.Axes, *, kpi: str, st: Dict[str, Any], unit: Optional[str], outlier_note: str) -> None:
    ax.axis("off")

    title = f"{kpi}" + (f" ({unit})" if unit else "")
    ax.text(0.0, 1.0, title, ha="left", va="top", fontsize=10, fontweight="bold")

    rows = [
        ("n", _format_num(st.get("n"))),
        ("missing", f"{float(st.get('missing_rate', 0.0)):.1%}"),
        ("mean", _format_num(st.get("mean"))),
        ("median", _format_num(st.get("median"))),
        ("std", _format_num(st.get("std"))),
        ("p05", _format_num(st.get("p05"))),
        ("p25", _format_num(st.get("p25"))),
        ("p75", _format_num(st.get("p75"))),
        ("p95", _format_num(st.get("p95"))),
        ("min", _format_num(st.get("min"))),
        ("max", _format_num(st.get("max"))),
    ]

    y0 = 0.86
    dy = 0.070
    for i, (k, v) in enumerate(rows):
        y = y0 - i * dy
        if y < 0.10:
            break
        ax.text(0.0, y, str(k), ha="left", va="top", fontsize=9, alpha=0.9)
        ax.text(0.98, y, str(v), ha="right", va="top", fontsize=9, alpha=0.9)

    if outlier_note:
        ax.text(0.0, 0.05, outlier_note, ha="left", va="bottom", fontsize=8, alpha=0.85)


def _pick_kpis(df: pd.DataFrame, resolved_y: Optional[str], max_kpis: int = 4) -> List[str]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(str(c))]
    if not num_cols:
        return []

    out: List[str] = []
    if isinstance(resolved_y, str) and resolved_y in num_cols:
        out.append(resolved_y)

    for c in num_cols:
        if c not in out:
            out.append(c)
        if len(out) >= max_kpis:
            break
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_stats_summary(*, df: pd.DataFrame, viz_spec: Dict[str, Any], aggregate_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    resolved = (viz_spec or {}).get("resolved") or {}
    resolved_y = resolved.get("y")
    resolved_group = resolved.get("group")

    kpis = _pick_kpis(df, resolved_y if isinstance(resolved_y, str) else None, max_kpis=4)
    if not kpis:
        return {"text": "No numeric KPI column found for stats summary.", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    group = _pick_group(df, resolved_group if isinstance(resolved_group, str) else None, kpis[0] if kpis else None)

    nrows = len(kpis)
    fig = plt.figure(figsize=(15.2, max(4.6, 2.8 * nrows)))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=3,
        width_ratios=[5.0, 1.2, 2.6],
        hspace=0.35,
        wspace=0.30,
    )

    viz_spec_ax = dict(viz_spec or {})
    if isinstance(viz_spec_ax.get("title"), str):
        viz_spec_ax["title"] = ""
    if isinstance(viz_spec_ax.get("subtitle"), str):
        viz_spec_ax["subtitle"] = ""

    for i, y in enumerate(kpis):
        ax = fig.add_subplot(gs[i, 0])
        ax_side = fig.add_subplot(gs[i, 1])
        ax_card = fig.add_subplot(gs[i, 2])

        df2 = df.copy()
        if group and group in df2.columns and (
            pd.api.types.is_object_dtype(df2[group]) or pd.api.types.is_string_dtype(df2[group])
        ):
            df2[group] = df2[group].astype("string").str.strip()

        s_all = pd.to_numeric(df2[y], errors="coerce").dropna()
        st = _stats(df2[y])
        unit = _unit_for(viz_spec, y)

        if s_all.empty:
            ax.axis("off")
            ax_side.axis("off")
            _make_stats_card(ax_card, kpi=str(y), st=st, unit=unit, outlier_note="No numeric data.")
            continue

        out_pol = (viz_spec or {}).get("outlier_policy") or {}
        b = infer_inlier_bounds(
            s_all,
            k=float(out_pol.get("iqr_k", 1.5)),
            min_n=int(out_pol.get("min_n", 30)),
        )
        low, high, s_in, _s_out = _inlier_view_range(s_all, b)

        if group and group in df2.columns and 1 < int(df2[group].nunique(dropna=True)) <= 20:
            gb = []
            labels = []
            for g, sub in df2.groupby(group, dropna=False):
                vals = pd.to_numeric(sub[y], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                gb.append(vals.values)
                labels.append(str(g))

            if gb:
                ax.boxplot(gb, labels=labels, showfliers=False)
                _quiet_axes(ax)

                apply_viz(ax=ax, df=df2, viz_spec=viz_spec_ax, x=group, y=None, group=None)
                ax.set_xlabel((viz_spec.get("x") or {}).get("label") or group or "Group")

                ycfg = (viz_spec.get("y") or {})
                y_label = (ycfg.get("label") or y).strip()
                unit2 = (viz_spec.get("units") or {}).get(y) or ycfg.get("unit")
                if unit2:
                    y_label = f"{y_label} ({unit2})"
                ax.set_ylabel(y_label)

                _make_side_full_range_box(ax_side, s_all.values)

                outlier_note = ""
                if bool(b.get("ok", False)):
                    outlier_note = "Fliers hidden in left plot; full-range shown in middle."
                _make_stats_card(ax_card, kpi=str(y), st=st, unit=unit, outlier_note=outlier_note)
                continue

        edges = _hist_edges_for_view(s_in.to_numpy(dtype=float), low, high)
        ax.hist(s_in.values, bins=edges, rwidth=0.98)
        _quiet_axes(ax)
        ax.set_ylabel("Count")

        df_viz = df2.copy()
        df_viz[y] = pd.to_numeric(df_viz[y], errors="coerce")
        df_viz = df_viz[df_viz[y].between(low, high)]
        apply_viz(ax=ax, df=df_viz if not df_viz.empty else df2, viz_spec=viz_spec_ax, x=y, y=None, group=None)

        _draw_center_lines_with_labels(ax, s_in)

        below = int((s_all < low).sum())
        above = int((s_all > high).sum())
        _annotate_outliers_hidden(ax, below=below, above=above)

        _make_side_full_range_box(ax_side, s_all.values)

        outlier_note = ""
        if below > 0 or above > 0:
            outlier_note = f"Left plot shows typical range; {below} below, {above} above shown as outliers (middle)."
        _make_stats_card(ax_card, kpi=str(y), st=st, unit=unit, outlier_note=outlier_note)

    title = (viz_spec or {}).get("title") or ("Stats summary" if len(kpis) == 1 else "Stats summaries")
    subtitle = ""
    if len(kpis) > 1:
        subtitle = f"Showing {len(kpis)} KPI(s) (max=4)."
    fig.suptitle(f"{title}\n{subtitle}".strip(), y=0.99)

    y0 = kpis[0]
    text = f"Plotted distribution-style view + stats summary for {len(kpis)} numeric KPI(s)."

    plan = aggregate_ctx.get("plan") if isinstance(aggregate_ctx, dict) else None
    if not isinstance(plan, dict):
        plan = aggregate_ctx if isinstance(aggregate_ctx, dict) else {}
    gran = plan.get("time_granularity")

    return {
        "text": text,
        "figures": [fig],
        "x": None,
        "y": y0,
        "group": group,
        "granularity": gran,
    }