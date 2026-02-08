# analysis/diagnostic/variance_decomposition.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
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


def _candidate_dimensions(df: pd.DataFrame, *, y: str, max_dims: int = 12) -> List[str]:
    dims: List[str] = []
    for c in df.columns:
        if c == y or _is_id_like(c):
            continue
        s = df[c]
        if _is_categorical_like(s):
            nun = int(s.nunique(dropna=True))
            if 2 <= nun <= 30:
                dims.append(c)
    return dims[:max_dims]


def _axis_pad_max(v: float, frac: float = 0.06, min_abs: float = 5.0) -> float:
    if not np.isfinite(v):
        return v
    pad = max(abs(v) * frac, min_abs * frac)
    return v + pad


def _get_cfg(viz_spec: Dict[str, Any]) -> Dict[str, Any]:
    cfg = (viz_spec or {}).get("diagnostic_config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault("min_group_n", 5)
    cfg.setdefault("shrinkage_k", 20.0)
    cfg.setdefault("top_k", 10)
    return cfg


def _shrink_mean(mean: float, n: float, overall_mean: float, k: float) -> float:
    denom = float(n + k)
    if denom <= 0:
        return float(mean)
    return float((n * mean + k * overall_mean) / denom)


def _pick_support_col(df: pd.DataFrame) -> Optional[str]:
    # Prefer pipeline's support column names
    for c in ["n_records", "count", "n"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _looks_mean_metric(y: str) -> bool:
    yl = (y or "").lower()
    return yl.startswith("mean_") or yl.startswith("avg_")


def run_variance_decomposition(*, df: pd.DataFrame, viz_spec: Dict[str, Any], aggregate_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Contribution / variance decomposition (Pareto drivers), stability-aware.

    Base contribution idea:
      contribution = n * (mean(group) - overall_mean)^2

    Best-practice stabilization:
      - Exclude "low support" groups from being ranked as top drivers (n < min_group_n)
      - Shrink small-n means toward overall mean using shrinkage_k before computing contribution

    Plot: Pareto chart
      bars = share (%) of (stability-adjusted) contribution
      line = cumulative share (%)

    Works for:
      - raw df (computes group means internally)
      - aggregated df (mean_* + count/n_records) from your pipeline
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    cfg = _get_cfg(viz_spec)
    min_group_n = int(cfg["min_group_n"])
    shrinkage_k = float(cfg["shrinkage_k"])
    top_k = int(cfg["top_k"])

    resolved = (viz_spec or {}).get("resolved") or {}
    y = _pick_numeric_y(df, resolved.get("y"))
    if not y or y not in df.columns:
        return {"text": "No numeric KPI column found for variance decomposition.", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    df_plot = df.copy()
    s = pd.to_numeric(df_plot[y], errors="coerce")
    df_plot = df_plot.loc[s.notna()].copy()
    if df_plot.empty:
        return {"text": f"No numeric values available in '{y}' for decomposition.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    dims = _candidate_dimensions(df_plot, y=y, max_dims=12)
    if not dims:
        return {"text": f"No suitable categorical dimensions (2–30 unique values) to explain variation in '{y}'.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    # detect aggregated-like input more robustly: needs a support column + mean-like metric
    support_col = _pick_support_col(df_plot)
    aggregated_like = bool(support_col is not None and _looks_mean_metric(str(y)))

    records: List[Dict[str, Any]] = []

    if aggregated_like:
        w = pd.to_numeric(df_plot[support_col], errors="coerce").fillna(0.0).astype(float)
        yv = pd.to_numeric(df_plot[y], errors="coerce").astype(float)

        wsum = float(w.sum())
        if wsum <= 0:
            return {"text": "Counts are missing/zero; cannot compute weighted contributions.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

        overall_mean = float((w * yv).sum() / wsum)

        # label using available dims in df (could be multiple dims because your aggregate grouped by multiple columns)
        label_parts = []
        for d in dims:
            label_parts.append(df_plot[d].astype("string").str.strip().fillna("(missing)").astype(str))
        if len(label_parts) == 1:
            driver_val = label_parts[0]
            dim_name = dims[0]
        else:
            driver_val = label_parts[0]
            for p in label_parts[1:]:
                driver_val = driver_val + " | " + p
            dim_name = " & ".join(dims)

        mean_shrunk = np.array(
            [_shrink_mean(float(m), float(n), overall_mean, shrinkage_k) for m, n in zip(yv.values, w.values)],
            dtype=float,
        )
        contrib = w.values * (mean_shrunk - overall_mean) ** 2

        tmp = pd.DataFrame(
            {
                "dimension": dim_name,
                "value": driver_val,
                "n": w.astype(float),
                "mean": yv.astype(float),
                "mean_shrunk": mean_shrunk,
                "contribution": contrib.astype(float),
            }
        )
        records = tmp.to_dict("records")
        basis_note = "Stability-aware variance decomposition (shrinkage)."
        N_basis = int(wsum)

    else:
        overall_mean = float(pd.to_numeric(df_plot[y], errors="coerce").mean())
        basis_note = "Stability-aware variance decomposition (shrinkage)."
        N_basis = int(len(df_plot))

        for dim in dims:
            d = df_plot[dim]
            if pd.api.types.is_object_dtype(d) or pd.api.types.is_string_dtype(d):
                d = d.astype("string").str.strip()

            tmp = df_plot.assign(**{dim: d}).groupby(dim, dropna=False)[y].agg(["count", "mean"]).reset_index()
            tmp.rename(columns={dim: "value", "count": "n"}, inplace=True)
            tmp["dimension"] = dim

            # If the input already contains per-group support (e.g., n_records), use it instead of row-count.
            # This fixes the "n=1" ticks when df_plot is already aggregated.
            if support_col is not None:
                try:
                    supp = df_plot.assign(**{dim: d}).groupby(dim, dropna=False)[support_col].sum().reset_index()
                    supp.rename(columns={dim: "value", support_col: "n_support"}, inplace=True)
                    tmp = tmp.merge(supp, on="value", how="left")
                    tmp["n"] = pd.to_numeric(tmp["n_support"], errors="coerce").fillna(tmp["n"]).astype(float)
                    tmp.drop(columns=["n_support"], inplace=True)
                except Exception:
                    pass

            tmp["mean_shrunk"] = tmp.apply(
                lambda r: _shrink_mean(float(r["mean"]), float(r["n"]), overall_mean, shrinkage_k),
                axis=1,
            )
            tmp["contribution"] = tmp["n"].astype(float) * (tmp["mean_shrunk"].astype(float) - overall_mean) ** 2
            records.extend(tmp[["dimension", "value", "n", "mean", "mean_shrunk", "contribution"]].to_dict("records"))

    contrib_df = pd.DataFrame(records)
    contrib_df = contrib_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["contribution"])
    contrib_df = contrib_df[contrib_df["contribution"] > 0].copy()

    if contrib_df.empty:
        return {
            "text": f"Could not compute meaningful contributions for '{y}' (insufficient variation after cleaning).",
            "figures": [],
            "x": None,
            "y": y,
            "group": None,
            "granularity": None,
        }

    contrib_df["low_support"] = contrib_df["n"].astype(float) < float(min_group_n)

    # ---- Labels: remove "dimension=" prefix from ticks; show only category (value) + n
    contrib_df["driver"] = contrib_df["value"].astype(str)
    contrib_df["driver_n"] = contrib_df["driver"].astype(str) + " (n=" + contrib_df["n"].astype(int).astype(str) + ")"

    stable = contrib_df.loc[~contrib_df["low_support"]].copy()
    low = contrib_df.loc[contrib_df["low_support"]].copy()

    if stable.empty:
        stable = contrib_df.copy()
        low = contrib_df.iloc[0:0].copy()
        low_support_note = "All groups are low-support under min_group_n; showing them anyway."
    else:
        low_support_note = f"Groups with n<{min_group_n} are not ranked as top drivers (bucketed separately)."

    stable = stable.sort_values("contribution", ascending=False)

    total = float(contrib_df["contribution"].sum())
    stable["share"] = stable["contribution"] / total

    top = stable.head(max(1, min(top_k, len(stable)))).copy()
    rest = stable.iloc[len(top):].copy()

    rows_for_plot: List[Dict[str, Any]] = []
    rows_for_plot.extend(top.to_dict("records"))

    if not rest.empty:
        rows_for_plot.append(
            {
                "driver_n": "(other stable drivers)",
                "contribution": float(rest["contribution"].sum()),
                "share": float(rest["contribution"].sum() / total),
            }
        )

    if not low.empty:
        rows_for_plot.append(
            {
                "driver_n": f"(low-support drivers: n<{min_group_n})",
                "contribution": float(low["contribution"].sum()),
                "share": float(low["contribution"].sum() / total),
            }
        )

    plot_df = pd.DataFrame(rows_for_plot)
    plot_df["share_pct"] = plot_df["share"].astype(float) * 100.0
    plot_df["cum_share_pct"] = plot_df["share_pct"].cumsum()

    # ---- Plot: Pareto ----
    fig = plt.figure(figsize=(11, 5.4))
    ax1 = fig.add_subplot(111)

    labels = plot_df["driver_n"].astype(str).tolist()
    x = np.arange(len(labels))

    ax1.bar(x, plot_df["share_pct"].astype(float).values)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    ax1.set_ylabel("Contribution share (%)")
    ax1.set_xlabel("Category, with support n")

    ax2 = ax1.twinx()
    ax2.plot(x, plot_df["cum_share_pct"].astype(float).values, marker="o")
    ax2.set_ylabel("Cumulative share (%)")

    ax1.set_ylim(0.0, _axis_pad_max(float(plot_df["share_pct"].max()), frac=0.10, min_abs=5.0))
    ax2.set_ylim(0.0, 105.0)
    ax2.axhline(80.0, linestyle="--", linewidth=1.0, alpha=0.7)

    title = (viz_spec or {}).get("title") or "Pareto drivers of KPI variation (stability-aware)"
    subtitle = f"Dimension: {', '.join(dims)} | KPI: {y} | overall mean≈{overall_mean:.3g} | basis N≈{N_basis} | shrinkage_k={shrinkage_k}"
    ax1.set_title(f"{title}\n{subtitle}")

    # ---- Textbox: ONLY procedure name (as requested)
    ax1.text(
        0.02,
        0.98,
        basis_note,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.65),
    )

    top_driver_txt = ""
    if not stable.empty:
        best = stable.iloc[0]
        top_driver_txt = f"Top stable driver: {best['value']} (n={int(best['n'])}, share≈{float(best['share']):.1%}). "

    text = (
        f"Computed stability-aware variance decomposition for '{y}' (screening only). "
        f"{top_driver_txt}"
        f"{low_support_note} {basis_note}"
    )

    return {"text": text, "figures": [fig], "x": "driver", "y": y, "group": None, "granularity": None}