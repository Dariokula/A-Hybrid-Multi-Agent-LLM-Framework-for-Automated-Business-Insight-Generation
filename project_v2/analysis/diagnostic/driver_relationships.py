# analysis/diagnostic/driver_relationships.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def _is_id_like(col: str) -> bool:
    cl = str(col).lower()
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
        cl = str(c).lower()
        if cl in {"n_records", "count", "n", "rows"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(c):
            return c
    return None


def _looks_aggregated_table(df: pd.DataFrame) -> bool:
    """Light guardrail: aggregated outputs often have counts or many mean_/sum_ style columns."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    if any(c in df.columns for c in ["n_records", "count"]):
        return True
    prefixes = ("mean_", "median_", "sum_", "min_", "max_", "avg_")
    pref_cols = [c for c in df.columns if str(c).lower().startswith(prefixes)]
    return (len(pref_cols) / max(1, len(df.columns))) >= 0.25


def _tufte_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.18)
    ax.grid(False, axis="x")


def _safe_std(x: pd.Series) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")
    v = float(s.std(ddof=0))
    return v if v != 0 else float("nan")


def _shrink_mean(mean: float, n: float, overall_mean: float, k: float) -> float:
    """
    James–Stein-like shrinkage toward overall mean:
      mean_shrunk = (n*mean + k*overall_mean) / (n+k)
    """
    denom = float(n + k)
    if denom <= 0:
        return float(mean)
    return float((n * mean + k * overall_mean) / denom)


def _pick_support_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["n_records", "count", "n"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _direction_legend_text() -> str:
    return "positive → increases KPI\nnegative → decreases KPI"


def _add_direction_legend(ax: plt.Axes) -> None:
    ax.text(
        0.02, 0.02,
        _direction_legend_text(),
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.55),
        clip_on=True,
    )


# -----------------------------
# Metrics
# -----------------------------
def _numeric_effects(
    df: pd.DataFrame,
    *,
    y: str,
    max_features: int = 8,
    min_n: int = 40,
) -> pd.DataFrame:
    """
    Numeric metric: Δy per +1 SD(x) in y-units.
      effect = corr(x, y) * sd(y)
    Direction: sign(effect)
    """
    rows: List[Dict[str, Any]] = []

    yv = pd.to_numeric(df[y], errors="coerce")
    y_sd = _safe_std(yv)
    if not np.isfinite(y_sd):
        return pd.DataFrame([])

    for c in df.columns:
        if c == y or _is_id_like(c):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        xv = pd.to_numeric(df[c], errors="coerce")
        m = xv.notna() & yv.notna()
        n = int(m.sum())
        if n < min_n:
            continue

        x_sd = _safe_std(xv[m])
        if not np.isfinite(x_sd):
            continue

        corr = float(pd.concat([xv[m], yv[m]], axis=1).corr().iloc[0, 1])
        if not np.isfinite(corr):
            continue

        effect = corr * float(y_sd)  # Δy per +1 SD(x), in y units
        rows.append(
            {
                "feature": c,
                "effect": float(effect),
                "abs_effect": float(abs(effect)),
                "corr": float(corr),
                "n": n,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("abs_effect", ascending=False).head(max_features).copy()
    out["label"] = out["feature"].astype(str) + " (n=" + out["n"].astype(int).astype(str) + ")"
    return out


def _categorical_lifts(
    df: pd.DataFrame,
    *,
    y: str,
    max_items: int = 10,
    min_group_n: int = 10,
    shrinkage_k: float = 20.0,
    max_levels_per_col: int = 12,
) -> pd.DataFrame:
    """
    Categorical metric: lift vs overall mean (in y units), stabilized by shrinkage.
      lift = mean_shrunk(category) - overall_mean

    We return the top (col=value) pairs by |lift|.
    """
    yv = pd.to_numeric(df[y], errors="coerce")
    m0 = yv.notna()
    if int(m0.sum()) < 30:
        return pd.DataFrame([])

    support_col = _pick_support_col(df)
    if support_col is not None:
        w = pd.to_numeric(df[support_col], errors="coerce").fillna(0.0)
        wsum = float(w[m0].sum())
        if wsum > 0:
            overall_mean = float((w[m0] * yv[m0]).sum() / wsum)
        else:
            overall_mean = float(yv[m0].mean())
    else:
        overall_mean = float(yv[m0].mean())

    rows: List[Dict[str, Any]] = []

    for c in df.columns:
        if c == y or _is_id_like(c):
            continue
        if not _is_categorical_like(df[c]):
            continue

        s = df[c].astype("string").str.strip()
        nun = int(s.nunique(dropna=True))
        if not (2 <= nun <= 30):
            continue

        vc = s.value_counts(dropna=False)
        keep = list(vc.index[:max_levels_per_col])
        s2 = s.where(s.isin(keep), "(other)")

        tmp = pd.DataFrame({"cat": s2, "y": yv})
        tmp = tmp.loc[tmp["y"].notna()].copy()

        if support_col is not None:
            tmp["w"] = pd.to_numeric(df.loc[tmp.index, support_col], errors="coerce").fillna(0.0)
            grp = tmp.groupby("cat", dropna=False).agg(n=("w", "sum"), mean=("y", "mean")).reset_index()
        else:
            grp = tmp.groupby("cat", dropna=False).agg(n=("y", "count"), mean=("y", "mean")).reset_index()

        for _, r in grp.iterrows():
            n = float(r["n"])
            if n < float(min_group_n):
                continue
            mean = float(r["mean"])
            mean_shrunk = _shrink_mean(mean, n, overall_mean, shrinkage_k)
            lift = mean_shrunk - overall_mean

            rows.append(
                {
                    "feature": c,
                    "value": str(r["cat"]),
                    "n": float(n),
                    "mean": mean,
                    "mean_shrunk": mean_shrunk,
                    "lift": float(lift),
                    "abs_lift": float(abs(lift)),
                    "overall_mean": overall_mean,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("abs_lift", ascending=False).head(max_items).copy()
    out["label"] = (
        out["feature"].astype(str)
        + "="
        + out["value"].astype(str)
        + " (n="
        + out["n"].astype(int).astype(str)
        + ")"
    )
    return out


# -----------------------------
# Main
# -----------------------------
def run_driver_relationships(
    *,
    df: pd.DataFrame,
    viz_spec: Dict[str, Any],
    aggregate_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compact, directional driver view (screening, not causality).

    Numeric drivers:
      Δy per +1 SD(x)  (in y-units)
      - sign: direction
      - magnitude: typical impact size

    Categorical drivers:
      Lift vs overall mean (in y-units)
      - sign: above/below average
      - magnitude: typical deviation from overall mean

    Output:
      Single figure with 2 panels side-by-side.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    resolved = (viz_spec or {}).get("resolved") or {}
    y = _pick_numeric_y(df, resolved.get("y"))
    if not y or y not in df.columns:
        return {"text": "No numeric KPI column found for driver relationships.", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    df2 = df.copy()
    yv = pd.to_numeric(df2[y], errors="coerce")
    df2 = df2.loc[yv.notna()].copy()
    if df2.empty:
        return {"text": f"No numeric values available in '{y}' for driver relationships.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    cfg = (viz_spec or {}).get("diagnostic_config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    top_k_num = int(cfg.get("top_k_numeric", 8))
    top_k_cat = int(cfg.get("top_k_categorical", 10))
    min_n_num = int(cfg.get("min_n_numeric", 40))
    min_group_n = int(cfg.get("min_group_n", 10))
    shrinkage_k = float(cfg.get("shrinkage_k", 20.0))

    # Most absolute influence selection happens inside these functions (abs_effect / abs_lift).
    num_df = _numeric_effects(df2, y=y, max_features=top_k_num, min_n=min_n_num)
    cat_df = _categorical_lifts(
        df2, y=y, max_items=top_k_cat, min_group_n=min_group_n, shrinkage_k=shrinkage_k, max_levels_per_col=12
    )

    agg_note = ""
    if _looks_aggregated_table(df2):
        agg_note = (
            "Note: input looks aggregated (e.g., mean_* columns or support counts). "
            "Directional drivers are usually most reliable on row-level data."
        )

    n = int(len(num_df)) if isinstance(num_df, pd.DataFrame) else 0
    m = int(len(cat_df)) if isinstance(cat_df, pd.DataFrame) else 0

    # ---------- Plot (2 panels) ----------
    fig, (axL, axR) = plt.subplots(
        1, 2,
        figsize=(13.2, 5.0),
        gridspec_kw={"width_ratios": [1.05, 1.15]},
        constrained_layout=True,
    )

    # Left: numeric effects
    _tufte_axes(axL)
    axL.set_title(f"Numerical drivers: top {n} drivers", fontsize=10, pad=10)
    if num_df.empty:
        axL.text(0.5, 0.5, "No numerical drivers found (insufficient data or no variance).", ha="center", va="center", fontsize=10)
        axL.set_xticks([])
        axL.set_yticks([])
    else:
        d = num_df.sort_values("abs_effect", ascending=True).copy()
        axL.barh(d["label"].tolist(), d["effect"].astype(float).values)
        axL.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
        axL.set_xlabel(f"Effect in {y} units (directional)")
        _add_direction_legend(axL)

    # Right: categorical lifts
    _tufte_axes(axR)
    axR.set_title(f"Categorical drivers: top {m} drivers", fontsize=10, pad=10)
    if cat_df.empty:
        axR.text(0.5, 0.5, "No categorical drivers found (no suitable categories or low support).", ha="center", va="center", fontsize=10)
        axR.set_xticks([])
        axR.set_yticks([])
    else:
        d = cat_df.sort_values("abs_lift", ascending=True).copy()
        axR.barh(d["label"].tolist(), d["lift"].astype(float).values)
        axR.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
        axR.set_xlabel(f"Lift in {y} units (above/below average)")
        _add_direction_legend(axR)

    # ---------- Compact text ----------
    overall_mean = float(pd.to_numeric(df2[y], errors="coerce").mean())
    overall_med = float(pd.to_numeric(df2[y], errors="coerce").median())

    parts = [
        f"Directional driver screening for '{y}' (no causal claims).",
        f"Overall: mean≈{overall_mean:.3g}, median≈{overall_med:.3g}.",
    ]

    if not num_df.empty:
        b = num_df.iloc[0]
        parts.append(
            f"Top numerical driver: {b['feature']} (Δ{y} per +1 SD ≈ {float(b['effect']):+.3g}, n={int(b['n'])})."
        )

    if not cat_df.empty:
        c0 = cat_df.iloc[0]
        parts.append(
            f"Top categorical lift: {c0['feature']}={c0['value']} (lift≈{float(c0['lift']):+.3g}, n={int(c0['n'])})."
        )

    if agg_note:
        parts.append(agg_note)

    text = " ".join(parts)

    return {"text": text, "figures": [fig], "x": None, "y": y, "group": None, "granularity": None}