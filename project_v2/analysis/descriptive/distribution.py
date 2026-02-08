# analysis/descriptive/distribution.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
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


def _pick_y(df: pd.DataFrame, resolved_y: Optional[str]) -> Optional[str]:
    if isinstance(resolved_y, str) and resolved_y in df.columns:
        return resolved_y
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if _is_categorical_like(df[c]):
            return c
    return None


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
    If outlier trimming collapses the inlier view to a single unique value while the
    full data has >1 unique value, relax bounds to show at least two classes when possible.

    Strategy:
      1) For integer-like data, round outward: low=floor(low), high=ceil(high)
      2) If still singleton, expand to include nearest distinct value(s) adjacent to the
         singleton value, preferring sides where data exists.
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

    # 1) If integer-like, round bounds outward to avoid slicing off integer bins.
    if _integer_like(vals_all):
        low2 = float(np.floor(low))
        high2 = float(np.ceil(high))
        s_in2 = s_all[(s_all >= low2) & (s_all <= high2)]
        if int(s_in2.nunique(dropna=True)) > 1:
            return low2, high2, s_in2
        low, high, s_in = low2, high2, s_in2

    # 2) Still singleton: include nearest distinct value(s).
    #    Work with unique sorted finite values.
    uniq = np.unique(vals_all)
    if uniq.size <= 1:
        return low, high, s_in

    # Use the inlier singleton value as "anchor" (closest to current interval mid if needed).
    in_vals = s_in.to_numpy(dtype=float)
    in_vals = in_vals[np.isfinite(in_vals)]
    if in_vals.size == 0:
        return low, high, s_in

    # anchor value: the (only) unique value in inliers
    anchor = float(np.unique(in_vals)[0])

    # Find neighbors around anchor
    idx = int(np.searchsorted(uniq, anchor))
    # searchsorted can return position between values; find exact match if present
    if idx < uniq.size and np.isclose(uniq[idx], anchor):
        anchor_idx = idx
    elif idx > 0 and np.isclose(uniq[idx - 1], anchor):
        anchor_idx = idx - 1
    else:
        # fallback: pick closest unique value as anchor
        anchor_idx = int(np.argmin(np.abs(uniq - anchor)))
        anchor = float(uniq[anchor_idx])

    below_val = float(uniq[anchor_idx - 1]) if anchor_idx - 1 >= 0 else None
    above_val = float(uniq[anchor_idx + 1]) if anchor_idx + 1 < uniq.size else None

    # Decide expansion direction(s)
    has_below_data = below_val is not None
    has_above_data = above_val is not None

    # Prefer expanding toward where we currently have outliers, if any
    below_out = int((s_all < low).sum())
    above_out = int((s_all > high).sum())

    low_new, high_new = low, high
    if below_out > 0 and has_below_data:
        low_new = min(low_new, below_val)
    if above_out > 0 and has_above_data:
        high_new = max(high_new, above_val)

    # If no outliers count info helps (or bounds didn't change), expand to the nearest side
    if np.isclose(low_new, low) and np.isclose(high_new, high):
        # choose the closest neighbor by distance
        cand = []
        if has_below_data:
            cand.append(("below", abs(anchor - below_val), below_val))
        if has_above_data:
            cand.append(("above", abs(above_val - anchor), above_val))
        cand.sort(key=lambda x: x[1])
        if cand:
            side, _, v = cand[0]
            if side == "below":
                low_new = min(low_new, v)
            else:
                high_new = max(high_new, v)

    # Ensure valid interval
    if not np.isfinite(low_new) or not np.isfinite(high_new) or high_new < low_new:
        return low, high, s_in

    s_in3 = s_all[(s_all >= low_new) & (s_all <= high_new)]
    # Only accept if we actually increased variety (or at least increased count meaningfully)
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

    # Edge case fix: if trimming collapses to one unique value, relax bounds.
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

    # Prefer RIGHT side, but if the vertical center lines (median/mean) are on the right half,
    # move the box to the LEFT to reduce collision risk.
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
                        xs.append(float(xd[0]))  # axvline produces constant-x lines
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
    """Right panel: boxplot incl. fliers. Styled to avoid thick/colored artifacts."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        ax_side.text(0.5, 0.5, "No data", transform=ax_side.transAxes, ha="center", va="center", fontsize=9, alpha=0.8)
        ax_side.set_xticks([])
        ax_side.set_yticks([])
        _quiet_axes(ax_side)
        return

    # Explicit styling to prevent "fat colored bar" (median/box fill artifacts)
    ax_side.boxplot(
        [vals],
        showfliers=True,
        vert=False,
        widths=0.55,
        patch_artist=False,  # no filled box
        medianprops=dict(linewidth=1.1, alpha=0.9),
        whiskerprops=dict(linewidth=1.0, alpha=0.85),
        capprops=dict(linewidth=1.0, alpha=0.85),
        boxprops=dict(linewidth=1.0, alpha=0.85),
        flierprops=dict(markersize=3, alpha=0.65),
    )
    ax_side.set_yticks([])
    ax_side.set_title("Removed outliers", fontsize=9, pad=6)  # <-- changed title
    ax_side.grid(False)
    _quiet_axes(ax_side)


def run_distribution(*, df: pd.DataFrame, viz_spec: Dict[str, Any], aggregate_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data to plot (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    resolved = (viz_spec or {}).get("resolved") or {}
    y = _pick_y(df, resolved.get("y"))
    if not y or y not in df.columns:
        return {"text": "No suitable column found for distribution/composition.", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    group = _pick_group(df, resolved.get("group"), y)

    df2 = df.copy()
    if group and group in df2.columns and (
        pd.api.types.is_object_dtype(df2[group]) or pd.api.types.is_string_dtype(df2[group])
    ):
        df2[group] = df2[group].astype("string").str.strip()

    # ----------------------------
    # Numeric distribution
    # ----------------------------
    if pd.api.types.is_numeric_dtype(df2[y]):
        s_all = pd.to_numeric(df2[y], errors="coerce").dropna()
        if s_all.empty:
            return {"text": f"No numeric values available in '{y}' for distribution.", "figures": [], "x": None, "y": y, "group": group, "granularity": None}

        out_pol = viz_spec.get("outlier_policy") or {}
        b = infer_inlier_bounds(
            s_all,
            k=float(out_pol.get("iqr_k", 1.5)),
            min_n=int(out_pol.get("min_n", 30)),
        )

        low, high, s_in, s_out = _inlier_view_range(s_all, b)

        fig, (ax, ax_side) = plt.subplots(
            1,
            2,
            figsize=(12, 4.5),
            gridspec_kw={"width_ratios": [5.0, 1.2]},
            constrained_layout=True,
        )

        # Grouped: boxplots
        if group and group in df2.columns and 1 < int(df2[group].nunique(dropna=True)) <= 20:
            gb = []
            labels = []
            for g, sub in df2.groupby(group, dropna=False):
                vals = pd.to_numeric(sub[y], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                gb.append(vals.values)
                labels.append(str(g))

            if not gb:
                return {"text": f"No numeric values available in '{y}' by group for distribution.", "figures": [], "x": group, "y": y, "group": group, "granularity": None}

            ax.boxplot(gb, labels=labels, showfliers=False)
            _quiet_axes(ax)

            # avoid a second outlier note from apply_viz
            apply_viz(ax=ax, df=df2, viz_spec=viz_spec, x=group, y=None, group=None)
            ax.set_xlabel((viz_spec.get("x") or {}).get("label") or group or "Group")

            ycfg = (viz_spec.get("y") or {})
            y_label = (ycfg.get("label") or y).strip()
            unit = (viz_spec.get("units") or {}).get(y) or ycfg.get("unit")
            if unit:
                y_label = f"{y_label} ({unit})"
            ax.set_ylabel(y_label)

            _make_side_full_range_box(ax_side, s_all.values)

            text = f"Plotted boxplot distribution of '{y}' by '{group}' (fliers hidden; removed outliers summary on the right)."
            return {"text": text, "figures": [fig], "x": group, "y": y, "group": group, "granularity": None}

        # Histogram: inlier/typical range view
        edges = _hist_edges_for_view(s_in.to_numpy(dtype=float), low, high)
        ax.hist(s_in.values, bins=edges, rwidth=0.98)
        _quiet_axes(ax)
        ax.set_ylabel("Count")

        # Prevent apply_viz from generating its own outlier note: use inlier-only df for styling
        df_viz = df2.copy()
        df_viz[y] = pd.to_numeric(df_viz[y], errors="coerce")
        df_viz = df_viz[df_viz[y].between(low, high)]
        apply_viz(ax=ax, df=df_viz if not df_viz.empty else df2, viz_spec=viz_spec, x=y, y=None, group=None)

        _draw_center_lines_with_labels(ax, s_in)

        below = int((s_all < low).sum())
        above = int((s_all > high).sum())
        _annotate_outliers_hidden(ax, below=below, above=above)

        _make_side_full_range_box(ax_side, s_all.values)

        if below > 0 or above > 0:
            text = f"Plotted histogram for '{y}' (typical range; removed outliers summarized on the right)."
        else:
            text = f"Plotted histogram distribution for '{y}'."

        return {"text": text, "figures": [fig], "x": y, "y": y, "group": group, "granularity": None}

    # ----------------------------
    # Categorical composition
    # ----------------------------
    if _is_categorical_like(df2[y]):
        fig = plt.figure(figsize=(11, 4.5))
        ax = fig.add_subplot(111)

        s = df2[y].astype("string").str.strip()
        vc = s.value_counts(dropna=False)
        if len(vc) > 20:
            top = vc.iloc[:20]
            other = vc.iloc[20:].sum()
            vc = top.copy()
            vc.loc["(other)"] = other

        xcats = [str(i) for i in vc.index.tolist()]
        counts = vc.values.astype(float)
        ax.bar(xcats, counts)
        _quiet_axes(ax)

        apply_viz(ax=ax, df=df2.assign(**{y: s}), viz_spec=viz_spec, x=y, y=None, group=None)
        ax.set_ylabel("Count")
        ax.set_xlabel((viz_spec.get("x") or {}).get("label") or y)
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")

        total = float(counts.sum()) if counts.size else 0.0
        top_label = xcats[int(np.argmax(counts))] if counts.size else "—"
        top_share = (float(np.max(counts)) / total) if total > 0 else 0.0
        text = f"Plotted composition for '{y}' ({len(vc)} categories shown). Top category: {top_label} ({top_share:.1%})."

        return {"text": text, "figures": [fig], "x": y, "y": None, "group": group, "granularity": None}

    return {"text": f"Column '{y}' is not suitable for distribution/composition.", "figures": [], "x": None, "y": y, "group": group, "granularity": None}