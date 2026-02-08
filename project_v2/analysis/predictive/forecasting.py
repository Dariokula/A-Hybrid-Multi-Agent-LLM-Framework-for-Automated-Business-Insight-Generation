# analysis/predictive/forecasting.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from analysis.viz_apply import apply_viz


def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
    if "time_bucket" in df.columns:
        return "time_bucket"
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def _pick_numeric_y(df: pd.DataFrame, resolved_y: Optional[str]) -> Optional[str]:
    if isinstance(resolved_y, str) and resolved_y in df.columns and pd.api.types.is_numeric_dtype(df[resolved_y]):
        return resolved_y
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and str(c).lower() not in {"n_records", "count", "n"}:
            return c
    return None


def _make_supervised(ts: pd.Series, *, lags: List[int], roll_windows: List[int]) -> Tuple[pd.DataFrame, pd.Series]:
    s = pd.to_numeric(ts, errors="coerce").astype(float)
    X = pd.DataFrame(index=s.index)

    for L in lags:
        X[f"lag_{L}"] = s.shift(L)

    for w in roll_windows:
        X[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        X[f"roll_std_{w}"] = s.shift(1).rolling(w).std()

    y = s.copy()
    df_xy = X.join(y.rename("y"))
    df_xy = df_xy.dropna()
    y_out = df_xy.pop("y")
    return df_xy, y_out


def _make_features_only(ts: pd.Series, *, lags: List[int], roll_windows: List[int]) -> pd.DataFrame:
    s = pd.to_numeric(ts, errors="coerce").astype(float)
    X = pd.DataFrame(index=s.index)

    for L in lags:
        X[f"lag_{L}"] = s.shift(L)

    for w in roll_windows:
        X[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        X[f"roll_std_{w}"] = s.shift(1).rolling(w).std()

    return X


def _scaled_sizes(n: pd.Series, s_min: float = 25.0, s_max: float = 180.0) -> pd.Series:
    nn = pd.to_numeric(n, errors="coerce").fillna(0.0)
    if len(nn) == 0:
        return pd.Series([], dtype=float)
    lo = float(nn.min())
    hi = float(nn.max())
    if hi <= lo:
        return pd.Series([0.5 * (s_min + s_max)] * len(nn), index=nn.index)
    return s_min + (nn - lo) * (s_max - s_min) / (hi - lo)


def _infer_freq_from_aggregate_ctx(idx: pd.DatetimeIndex, aggregate_ctx: Dict[str, Any]) -> str:
    plan = {}
    if isinstance(aggregate_ctx, dict):
        plan = aggregate_ctx.get("plan") if isinstance(aggregate_ctx.get("plan"), dict) else aggregate_ctx
    gran = str((plan or {}).get("time_granularity") or "").lower().strip()

    if gran == "day":
        return "D"
    if gran == "week":
        return "W-MON"
    if gran == "month":
        return "MS"
    if gran == "quarter":
        return "QS"
    if gran == "year":
        return "YS"

    try:
        f = pd.infer_freq(idx)
        return f or "MS"
    except Exception:
        return "MS"


def _infer_granularity_from_aggregate_ctx(aggregate_ctx: Dict[str, Any]) -> Optional[str]:
    plan = {}
    if isinstance(aggregate_ctx, dict):
        plan = aggregate_ctx.get("plan") if isinstance(aggregate_ctx.get("plan"), dict) else aggregate_ctx
    gran = str((plan or {}).get("time_granularity") or "").lower().strip()
    return gran or None


def _default_horizon_for_granularity(gran: Optional[str]) -> int:
    """
    Default ONLY if prompt doesn't specify. This is not a "magic constant" in the sense
    of hardcoding user intent; it's an adaptive UI default based on frequency.
    """
    g = (gran or "").lower()
    if g == "day":
        return 30
    if g == "week":
        return 12
    if g == "month":
        return 6
    if g == "quarter":
        return 6
    if g == "year":
        return 3
    return 6


def _steps_from_amount(amount: int, unit: str, gran: Optional[str]) -> int:
    """
    Convert a user horizon like "3 months" into number of steps of the series.
    We assume the series is already aggregated at granularity=gran.
    """
    g = (gran or "").lower()

    unit = unit.lower().strip()
    if unit in {"d", "day", "days", "tag", "tage"}:
        if g == "day":
            return amount
        if g == "week":
            return max(1, int(round(amount / 7.0)))
        if g == "month":
            return max(1, int(round(amount / 30.0)))
        if g == "quarter":
            return max(1, int(round(amount / 91.0)))
        if g == "year":
            return max(1, int(round(amount / 365.0)))

    if unit in {"w", "wk", "week", "weeks", "woche", "wochen"}:
        if g == "week":
            return amount
        if g == "day":
            return amount * 7
        if g == "month":
            return max(1, int(round(amount / 4.345)))
        if g == "quarter":
            return max(1, int(round(amount / 13.0)))
        if g == "year":
            return max(1, int(round(amount / 52.0)))

    if unit in {"m", "mon", "month", "months", "monat", "monate"}:
        if g == "month":
            return amount
        if g == "week":
            return max(1, int(round(amount * 4.345)))
        if g == "day":
            return max(1, int(round(amount * 30.0)))
        if g == "quarter":
            return max(1, int(round(amount / 3.0)))
        if g == "year":
            return max(1, int(round(amount / 12.0)))

    if unit in {"q", "quarter", "quarters", "quartal", "quartale"}:
        if g == "quarter":
            return amount
        if g == "month":
            return amount * 3
        if g == "week":
            return max(1, int(round(amount * 13.0)))
        if g == "day":
            return max(1, int(round(amount * 91.0)))
        if g == "year":
            return max(1, int(round(amount / 4.0)))

    if unit in {"y", "yr", "year", "years", "jahr", "jahre"}:
        if g == "year":
            return amount
        if g == "quarter":
            return amount * 4
        if g == "month":
            return amount * 12
        if g == "week":
            return amount * 52
        if g == "day":
            return amount * 365

    # fallback: interpret as steps directly
    return amount


def _infer_horizon_from_prompt(prompt: str, *, gran: Optional[str], max_horizon: int = 52) -> int:
    """
    Extract horizon from natural language:
      - "next 12 weeks", "for 3 months", "forecast 2 quarters", "1 year ahead"
      - "next month" (=> 1 month)
      - "until end of year" / "rest of the year" (approx based on granularity)
    """
    p = (prompt or "").lower()

    # Direct "horizon=12" or "12 periods"
    m = re.search(r"\bhorizon\s*[:=]\s*(\d{1,3})\b", p)
    if m:
        return int(np.clip(int(m.group(1)), 1, max_horizon))
    m = re.search(r"\b(\d{1,3})\s*(periods|steps|points)\b", p)
    if m:
        return int(np.clip(int(m.group(1)), 1, max_horizon))

    # "next N unit" / "for N unit" / "N unit ahead"
    m = re.search(r"\b(next|for|over|in|ahead|within)\s*(\d{1,3})\s*(days?|weeks?|months?|quarters?|years?|tage|wochen|monate?|quartale?|jahre?)\b", p)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        steps = _steps_from_amount(n, unit, gran)
        return int(np.clip(steps, 1, max_horizon))

    # "forecast N unit" / "predict N unit"
    m = re.search(r"\b(forecast|predict|projection|prognose)\s*(\d{1,3})\s*(days?|weeks?|months?|quarters?|years?|tage|wochen|monate?|quartale?|jahre?)\b", p)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        steps = _steps_from_amount(n, unit, gran)
        return int(np.clip(steps, 1, max_horizon))

    # "next month/quarter/year" without number
    if re.search(r"\bnext\s+month\b|\bnächsten\s+monat\b", p):
        return int(np.clip(_steps_from_amount(1, "month", gran), 1, max_horizon))
    if re.search(r"\bnext\s+quarter\b|\bnächstes\s+quartal\b", p):
        return int(np.clip(_steps_from_amount(1, "quarter", gran), 1, max_horizon))
    if re.search(r"\bnext\s+year\b|\bnächstes\s+jahr\b", p):
        return int(np.clip(_steps_from_amount(1, "year", gran), 1, max_horizon))

    # "rest of the year" / "until end of year" approximation
    if any(k in p for k in ["rest of the year", "until end of year", "end of the year", "bis jahresende", "rest des jahres", "jahresende"]):
        now = pd.Timestamp.now()
        end = pd.Timestamp(year=now.year, month=12, day=31)
        days = max(1, int((end - now).days))
        steps = _steps_from_amount(days, "days", gran)
        return int(np.clip(steps, 1, max_horizon))

    # No explicit request -> adaptive default
    return int(np.clip(_default_horizon_for_granularity(gran), 1, max_horizon))


def _recent_trend_slope(ts: pd.Series, window: int = 10) -> float:
    s = pd.to_numeric(ts, errors="coerce").dropna()
    if len(s) < 3:
        return 0.0
    w = min(window, len(s))
    y = s.iloc[-w:].to_numpy(dtype=float)
    x = np.arange(w, dtype=float)

    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad > 0:
        z = 0.6745 * (y - med) / mad
        keep = np.abs(z) <= 4.0
        if keep.sum() >= 3:
            y = y[keep]
            x = x[keep]

    if len(y) < 3:
        return 0.0

    slope = float(np.polyfit(x, y, 1)[0])

    recent_std = float(np.std(y)) if len(y) > 1 else 0.0
    if recent_std > 0:
        cap = 2.5 * recent_std
        slope = float(np.clip(slope, -cap, cap))
    return slope


def _blend_weight(step: int, horizon: int, start_w: float = 0.65, end_w: float = 0.20) -> float:
    if horizon <= 1:
        return start_w
    frac = (step - 1) / (horizon - 1)
    return float(start_w + (end_w - start_w) * frac)


def run_forecasting(
    *,
    df: pd.DataFrame,
    viz_spec: Dict[str, Any],
    aggregate_ctx: Dict[str, Any],
    prompt: str = "",
) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    resolved = (viz_spec or {}).get("resolved") or {}
    tcol = _pick_time_col(df)
    ycol = _pick_numeric_y(df, resolved.get("y"))

    if not tcol:
        return {"text": "No datetime/time column found for forecasting (need a time axis).", "figures": [], "x": None, "y": ycol, "group": None, "granularity": None}
    if not ycol:
        return {"text": "No numeric target column found for forecasting.", "figures": [], "x": tcol, "y": None, "group": None, "granularity": None}

    d = df.copy()
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d = d.dropna(subset=[tcol]).sort_values(tcol)

    d_clean = d.dropna(subset=[ycol]).copy()
    if len(d_clean) < 30:
        return {"text": f"Not enough time points for forecasting on '{ycol}' (need ~30+, got {len(d_clean)}).", "figures": [], "x": tcol, "y": ycol, "group": None, "granularity": None}

    support_col = "n_records" if "n_records" in d_clean.columns else None

    ts = d_clean.set_index(tcol)[ycol].astype(float)
    idx = ts.index
    if not isinstance(idx, pd.DatetimeIndex):
        return {"text": "Time axis is not a DatetimeIndex after parsing; cannot forecast into the future.", "figures": [], "x": tcol, "y": ycol, "group": None, "granularity": None}

    lags = [1, 2, 3, 4, 6, 8, 12]
    roll_windows = [3, 6, 12]
    X, y = _make_supervised(ts, lags=lags, roll_windows=roll_windows)

    if len(X) < 25:
        return {"text": f"Not enough usable rows after lag/rolling features for '{ycol}'.", "figures": [], "x": tcol, "y": ycol, "group": None, "granularity": None}

    n = len(X)
    split = max(10, int(0.8 * n))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    y_test_pred = pd.Series(model.predict(X_test), index=y_test.index, name="pred")
    mae = float(mean_absolute_error(y_test, y_test_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    # ✅ Horizon derived from prompt (with granularity-aware fallback)
    gran = _infer_granularity_from_aggregate_ctx(aggregate_ctx)
    horizon = _infer_horizon_from_prompt(prompt, gran=gran, max_horizon=52)

    freq = _infer_freq_from_aggregate_ctx(idx, aggregate_ctx)
    last_t = idx.max()
    future_idx = pd.date_range(start=last_t, periods=horizon + 1, freq=freq)[1:]

    slope = _recent_trend_slope(ts, window=10)
    last_val = float(ts.loc[last_t])

    hist = ts.copy()
    future_preds: List[Tuple[pd.Timestamp, float]] = []

    for step_i, t in enumerate(future_idx, start=1):
        if t not in hist.index:
            hist.loc[t] = np.nan

        X_all = _make_features_only(hist, lags=lags, roll_windows=roll_windows)
        if t not in X_all.index:
            break

        x_t = X_all.loc[[t]]
        if x_t.isna().any(axis=1).iloc[0]:
            break

        rf_pred = float(model.predict(x_t)[0])
        trend_pred = last_val + slope * float(step_i)

        w = _blend_weight(step_i, horizon, start_w=0.65, end_w=0.20)
        p = w * trend_pred + (1.0 - w) * rf_pred

        hist.loc[t] = p
        future_preds.append((t, p))

    future_series = pd.Series({t: p for t, p in future_preds}, name="forecast") if future_preds else None

    fig = plt.figure(figsize=(11, 4.8))
    ax = fig.add_subplot(111)

    ax.plot(ts.index, ts.values, linewidth=1.6, label="Actual")

    if support_col:
        sup = d_clean.set_index(tcol)[support_col].reindex(ts.index)
        ax.scatter(ts.index, ts.values, s=_scaled_sizes(sup), alpha=0.85)
        ax.text(
            0.99, 0.02,
            "Data point size implies amount of aggregated data",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8,
            alpha=0.8,
        )
    else:
        ax.scatter(ts.index, ts.values, s=45, alpha=0.85)

    if future_series is not None and not future_series.empty:
        fx = [last_t] + list(future_series.index)
        fy = [last_val] + list(future_series.values)
        ax.plot(fx, fy, marker="o", linewidth=1.8, label=f"Forecast (+{len(future_series)})")

    local_viz = dict(viz_spec or {})
    local_viz["title"] = local_viz.get("title") or "Forecast"
    local_viz["subtitle"] = local_viz.get("subtitle") or f"Target: {ycol} | MAE≈{mae:.3g} | RMSE≈{rmse:.3g} | Horizon={horizon}"
    local_viz.setdefault("x", {})
    local_viz.setdefault("y", {})
    local_viz["x"]["label"] = local_viz["x"].get("label") or tcol
    local_viz["y"]["label"] = local_viz["y"].get("label") or ycol

    if future_series is not None and not future_series.empty:
        df_for_viz = pd.DataFrame({tcol: list(ts.index) + list(future_series.index), ycol: list(ts.values) + list(future_series.values)})
    else:
        df_for_viz = pd.DataFrame({tcol: ts.index, ycol: ts.values})

    apply_viz(ax=ax, df=df_for_viz, viz_spec=local_viz, x=tcol, y=ycol, group=None)
    ax.legend(frameon=False)

    text = (
        f"Trained a forecasting model for '{ycol}' using lag/rolling features. "
        f"Test MAE≈{mae:.3g}, RMSE≈{rmse:.3g}. "
        f"Forecast horizon derived from prompt: {horizon} step(s) at freq='{freq}'."
    )

    return {"text": text, "figures": [fig], "x": tcol, "y": ycol, "group": None, "granularity": gran}