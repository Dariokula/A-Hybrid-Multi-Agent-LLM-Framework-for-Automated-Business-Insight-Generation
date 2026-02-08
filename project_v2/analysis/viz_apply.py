# analysis/viz_apply.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


def _series_is_integer_like(s: pd.Series, *, atol: float = 1e-9) -> bool:
    try:
        x = pd.to_numeric(s, errors="coerce")
        xv = x.dropna().astype(float)
        if xv.empty:
            return False
        return bool(np.allclose(xv.values, np.round(xv.values), atol=float(atol)))
    except Exception:
        return False


def _widen_sparse_bars(ax) -> None:
    try:
        patches = list(getattr(ax, "patches", []) or [])
        if not patches:
            return

        widths = [getattr(p, "get_width")() for p in patches]
        if not widths:
            return
        n = len(widths)
        if n > 12:
            return
        w_med = float(np.median([float(w) for w in widths if np.isfinite(w)]))
        if not (0.1 <= w_med <= 0.9):
            return

        if n <= 6:
            new_w = 0.98
        elif n <= 8:
            new_w = 0.95
        else:
            new_w = 0.90

        for p in patches:
            x0 = float(p.get_x())
            w0 = float(p.get_width())
            if not np.isfinite(x0) or not np.isfinite(w0):
                continue
            if w0 <= 0:
                continue
            if new_w <= w0:
                continue
            shift = (new_w - w0) / 2.0
            p.set_x(x0 - shift)
            p.set_width(new_w)

        try:
            ax.margins(x=0.02)
        except Exception:
            pass
    except Exception:
        return


def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _apply_spines(ax, spines: str):
    if spines == "left_bottom":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    elif spines == "none":
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        for s in ax.spines.values():
            s.set_visible(True)


def _apply_time_axis(ax, *, max_ticks: int, date_format: str = "auto"):
    loc = mdates.AutoDateLocator(minticks=3, maxticks=max(3, int(max_ticks)))
    ax.xaxis.set_major_locator(loc)
    if not date_format or date_format == "auto":
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    else:
        try:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        except Exception:
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _iqr_inlier_mask(s: pd.Series, *, k: float = 1.5) -> Tuple[pd.Series, Dict[str, Any]]:
    x = pd.to_numeric(s, errors="coerce")
    x_valid = x.dropna()
    if x_valid.empty:
        return pd.Series([False] * len(s), index=s.index), {"method": "iqr", "k": k, "ok": False}

    q1 = float(x_valid.quantile(0.25))
    q3 = float(x_valid.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        m = x.notna()
        return m, {"method": "iqr", "k": k, "ok": True, "q1": q1, "q3": q3, "iqr": iqr, "low": q1, "high": q3}

    low = q1 - k * iqr
    high = q3 + k * iqr
    m = x.notna() & (x >= low) & (x <= high)
    return m, {"method": "iqr", "k": k, "ok": True, "q1": q1, "q3": q3, "iqr": iqr, "low": low, "high": high}


def infer_inlier_bounds(
    s: pd.Series,
    *,
    k: float = 1.5,
    min_n: int = 30,
) -> Dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce")
    xv = x.dropna()
    if xv.empty or int(xv.shape[0]) < int(min_n):
        return {"ok": False, "n_valid": int(xv.shape[0]), "min": float(xv.min()) if not xv.empty else None, "max": float(xv.max()) if not xv.empty else None}

    m, stats = _iqr_inlier_mask(x, k=float(k))
    if not bool(stats.get("ok", False)):
        return {"ok": False, "n_valid": int(xv.shape[0]), "min": float(xv.min()), "max": float(xv.max())}

    low = float(stats.get("low"))
    high = float(stats.get("high"))
    out_mask = x.notna() & (~m)
    out_n = int(out_mask.sum())
    out_frac = out_n / max(1, int(xv.shape[0]))
    return {
        "ok": True,
        "n_valid": int(xv.shape[0]),
        "q1": float(stats.get("q1")) if stats.get("q1") is not None else None,
        "q3": float(stats.get("q3")) if stats.get("q3") is not None else None,
        "iqr": float(stats.get("iqr")) if stats.get("iqr") is not None else None,
        "low": low,
        "high": high,
        "outlier_count": out_n,
        "outlier_frac": float(out_frac),
        "min": float(xv.min()),
        "max": float(xv.max()),
    }


def prepare_plot_df(
    *,
    df: pd.DataFrame,
    viz_spec: Dict[str, Any],
    y: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or not y or y not in df.columns:
        return df, {"applied": False}

    pol = (viz_spec or {}).get("outlier_policy") or {}
    if not isinstance(pol, dict) or not bool(pol.get("enabled", True)):
        return df, {"applied": False}

    method = str(pol.get("method", "iqr")).lower()
    max_frac = float(pol.get("max_outlier_frac", 0.02))
    min_n = int(pol.get("min_n", 30))
    k = float(pol.get("iqr_k", 1.5))

    s = pd.to_numeric(df[y], errors="coerce")
    n_valid = int(s.notna().sum())
    if n_valid < min_n:
        return df, {"applied": False, "reason": f"n_valid<{min_n}"}

    if method != "iqr":
        return df, {"applied": False, "reason": f"unknown_method:{method}"}

    inlier_mask, stats = _iqr_inlier_mask(df[y], k=k)
    if not bool(stats.get("ok", False)):
        return df, {"applied": False, "reason": "iqr_failed"}

    out_mask = s.notna() & (~inlier_mask)
    out_n = int(out_mask.sum())
    if out_n == 0:
        return df, {"applied": False, "reason": "no_outliers"}

    out_frac = out_n / max(1, n_valid)
    if out_frac > max_frac:
        return df, {"applied": False, "reason": "outliers_not_rare", "outlier_count": out_n, "outlier_frac": out_frac}

    kept = df[inlier_mask].copy()
    out_vals = s[out_mask]
    kept_vals = s[inlier_mask]

    info = {
        "applied": True,
        "method": "iqr",
        "iqr_k": k,
        "max_outlier_frac": max_frac,
        "min_n": min_n,
        "n_valid": n_valid,
        "outlier_count": out_n,
        "outlier_frac": float(out_frac),
        "kept_min": float(kept_vals.min()) if not kept_vals.empty else None,
        "kept_max": float(kept_vals.max()) if not kept_vals.empty else None,
        "dropped_min": float(out_vals.min()) if not out_vals.empty else None,
        "dropped_max": float(out_vals.max()) if not out_vals.empty else None,
        "fence_low": float(stats.get("low")) if stats.get("low") is not None else None,
        "fence_high": float(stats.get("high")) if stats.get("high") is not None else None,
    }
    return kept, info


def _apply_axis_padding(ax, *, pad_frac: float, pad_min_rel: float):
    try:
        y0, y1 = ax.get_ylim()
    except Exception:
        return

    if not (np.isfinite(y0) and np.isfinite(y1)):
        return

    if y1 == y0:
        base = max(1.0, abs(y1))
        pad = max(pad_frac * base, pad_min_rel * base)
        ax.set_ylim(y0 - pad, y1 + pad)
        return

    rng = y1 - y0
    base = max(abs(y0), abs(y1), 1.0)
    pad = max(pad_frac * rng, pad_min_rel * base)
    ax.set_ylim(y0 - pad, y1 + pad)


def _annotate_excluded_outliers(ax, out_info: Dict[str, Any], *, y: str):
    if not isinstance(out_info, dict) or not out_info.get("applied", False):
        return

    oc = out_info.get("outlier_count")
    frac = out_info.get("outlier_frac")
    dmin = out_info.get("dropped_min")
    dmax = out_info.get("dropped_max")

    if oc is None or frac is None:
        return

    txt = f"Outliers excluded for viz: {int(oc)} ({float(frac):.1%})"
    if dmin is not None and dmax is not None:
        txt += f"\nExcluded {y} range: [{dmin:.4g}, {dmax:.4g}]"

    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.6),
    )


def _annotate_outliers_hidden(ax, *, x_below: int, x_above: int, y_below: int, y_above: int):
    xb = int(x_below)
    xa = int(x_above)
    yb = int(y_below)
    ya = int(y_above)
    if xb <= 0 and xa <= 0 and yb <= 0 and ya <= 0:
        return

    def _fmt(b: int, a: int) -> str:
        parts = []
        if b > 0:
            parts.append(f"{b} below")
        if a > 0:
            parts.append(f"{a} above")
        return ", ".join(parts) if parts else "0"

    if (yb > 0 or ya > 0) and (xb > 0 or xa > 0):
        detail = f"x-axis { _fmt(xb, xa) }; y-axis { _fmt(yb, ya) }"
    elif xb > 0 or xa > 0:
        detail = _fmt(xb, xa)
    else:
        detail = _fmt(yb, ya)

    txt = f"Outliers hidden from the graphic: {detail}."
    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.60),
    )


def apply_viz(
    *,
    ax,
    df: pd.DataFrame,
    viz_spec: Dict[str, Any],
    x: Optional[str],
    y: Optional[str],
    group: Optional[str] = None,
    outlier_info: Optional[Dict[str, Any]] = None,
):
    v = viz_spec or {}
    style = v.get("style") or {}
    legend = v.get("legend") or {}
    xcfg = v.get("x") or {}
    ycfg = v.get("y") or {}

    title = (v.get("title") or "").strip()
    subtitle = (v.get("subtitle") or "").strip()
    full_title = title if not subtitle else f"{title}\n{subtitle}".strip()
    if full_title:
        ax.set_title(full_title)

    units_map = (v.get("units") or {}) if isinstance(v.get("units"), dict) else {}

    # X label (+ unit support)
    if x:
        x_label = (xcfg.get("label") or x).strip()
        x_unit = units_map.get(x) or xcfg.get("unit", None)
        if x_unit:
            x_label = f"{x_label} ({x_unit})"
        ax.set_xlabel(x_label)

    # Y label (+ unit support)
    if y:
        y_label = (ycfg.get("label") or y).strip()
        y_unit = units_map.get(y) or ycfg.get("unit", None)
        if y_unit:
            y_label = f"{y_label} ({y_unit})"
        ax.set_ylabel(y_label)

    if bool(style.get("grid", True)):
        ax.grid(True, alpha=0.25)
    _apply_spines(ax, str(style.get("spines", "left_bottom") or "left_bottom"))

    axis_policy = v.get("axis_policy") or {}

    x_clip_below = 0
    x_clip_above = 0
    if x and x in df.columns:
        xpol = axis_policy.get("x") or {}
        is_time = bool(xpol.get("is_time", False))
        max_ticks = int(xpol.get("max_ticks", 10) or 10)
        rot = int(xpol.get("tick_rotation", xcfg.get("tick_rotation", 0) or 0) or 0)

        if is_time:
            _apply_time_axis(ax, max_ticks=max_ticks, date_format=str(xpol.get("date_format", "auto")))
        else:
            use_int = bool(pd.api.types.is_numeric_dtype(df[x])) and _series_is_integer_like(df[x])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max(3, max_ticks), integer=use_int))

            if pd.api.types.is_numeric_dtype(df[x]):
                lim = (xpol.get("limits") or {})
                xmin = lim.get("min", None)
                xmax = lim.get("max", None)
                if xmin is not None and xmax is not None and np.isfinite(xmin) and np.isfinite(xmax) and xmax != xmin:
                    xs = pd.to_numeric(df[x], errors="coerce")
                    x_clip_below = int((xs.notna() & (xs < xmin)).sum())
                    x_clip_above = int((xs.notna() & (xs > xmax)).sum())
                    rng = float(xmax - xmin)
                    pad = 0.03 * rng
                    ax.set_xlim(left=float(xmin - pad), right=float(xmax + pad))

        if rot:
            for t in ax.get_xticklabels():
                t.set_rotation(rot)
                t.set_ha("right")

    y_clip_below = 0
    y_clip_above = 0
    if y:
        ypol = axis_policy.get("y") or {}
        max_ticks = int(ypol.get("max_ticks", 6) or 6)
        scale = str(ypol.get("scale", "linear") or "linear").lower()

        if scale == "log":
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")

        lim = ypol.get("limits") or {}
        ymin = lim.get("min", None)
        ymax = lim.get("max", None)

        if ymin is not None or ymax is not None:
            ys = pd.to_numeric(df[y], errors="coerce") if (y in df.columns) else pd.Series(dtype="float64")
            if ymin is not None and np.isfinite(ymin):
                y_clip_below = int((ys.notna() & (ys < float(ymin))).sum())
            if ymax is not None and np.isfinite(ymax):
                y_clip_above = int((ys.notna() & (ys > float(ymax))).sum())
            ax.set_ylim(bottom=ymin, top=ymax)

        use_int = False
        try:
            if y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
                use_int = _series_is_integer_like(df[y])
        except Exception:
            use_int = False
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max(3, max_ticks), integer=bool(use_int)))

        pad_cfg = (v.get("axis_padding") or {})
        pad_frac = float(pad_cfg.get("pad_frac", 0.03))
        pad_min_rel = float(pad_cfg.get("pad_min_rel", 0.01))
        if scale != "log":
            _apply_axis_padding(ax, pad_frac=pad_frac, pad_min_rel=pad_min_rel)

        if outlier_info is not None and y:
            _annotate_excluded_outliers(ax, outlier_info, y=y)

    _annotate_outliers_hidden(ax, x_below=x_clip_below, x_above=x_clip_above, y_below=y_clip_below, y_above=y_clip_above)
    _widen_sparse_bars(ax)

    if bool(legend.get("show", True)) and group:
        loc = str(legend.get("loc", "best") or "best")
        ncol = int(legend.get("ncol", 1) or 1)
        ax.legend(loc=loc, ncol=ncol, frameon=False)