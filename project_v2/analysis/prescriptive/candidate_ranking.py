# analysis/prescriptive/candidate_ranking.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from analysis.viz_apply import apply_viz


def _is_id_like(col: str) -> bool:
    cl = col.lower()
    return any(k in cl for k in ["id", "uuid", "guid", "key", "nr", "no.", "number", "pos"])


def _pick_numeric_target(df: pd.DataFrame, prompt: str) -> Optional[str]:
    p = (prompt or "").lower()
    for c in df.columns:
        if c.lower() in p and pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(c):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_id_like(c):
            return c
    return None


def _objective_direction(prompt: str) -> str:
    p = (prompt or "").lower()
    if any(w in p for w in ["minimize", "reduce", "lower", "decrease", "shorten", "verringern", "senken", "reduzieren"]):
        return "minimize"
    if any(w in p for w in ["maximize", "increase", "raise", "improve", "steigern", "erhöhen", "maximieren"]):
        return "maximize"
    return "optimize"


def _pick_candidate_dimension(df: pd.DataFrame, prompt: str, target: str) -> Optional[str]:
    p = (prompt or "").lower()

    # if user mentions a categorical column name, pick it
    for c in df.columns:
        if c.lower() in p and c != target:
            s = df[c]
            if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s)) and 2 <= int(s.nunique(dropna=True)) <= 50:
                return c

    # fallback: a “nice” categorical dimension
    best = None
    best_score = -1
    for c in df.columns:
        if c == target or _is_id_like(c):
            continue
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
            nun = int(s.nunique(dropna=True))
            if 2 <= nun <= 50:
                # prefer actionable-like names
                name = c.lower()
                score = 0
                if any(k in name for k in ["resource", "group", "type", "status", "region", "product", "avg_type", "ta_"]):
                    score += 2
                score += 1 if nun <= 20 else 0
                if score > best_score:
                    best_score = score
                    best = c
    return best


def _prep_features(df: pd.DataFrame, y: str, *, max_levels: int = 20) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    d = df.copy()
    yv = pd.to_numeric(d[y], errors="coerce")
    d = d.loc[yv.notna()].copy()
    yv = pd.to_numeric(d[y], errors="coerce").astype(float)

    Xraw = d.drop(columns=[y])

    defaults: Dict[str, Any] = {}
    cat_levels: Dict[str, List[str]] = {}

    for c in list(Xraw.columns):
        if _is_id_like(c):
            Xraw = Xraw.drop(columns=[c])
            continue

        if pd.api.types.is_numeric_dtype(Xraw[c]):
            s = pd.to_numeric(Xraw[c], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            defaults[c] = med
            Xraw[c] = s.fillna(med)
        else:
            s = Xraw[c].astype("string").str.strip().fillna("(missing)")
            vc = s.value_counts(dropna=False)
            keep = vc.index[:max_levels].astype(str).tolist()
            cat_levels[c] = keep
            mode = str(vc.index[0]) if len(vc) else "(missing)"
            defaults[c] = mode
            Xraw[c] = s.where(s.astype(str).isin(set(keep)), "(other)")

    X = pd.get_dummies(Xraw, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta = {"defaults": defaults, "cat_levels": cat_levels, "raw_columns": list(Xraw.columns), "ohe_columns": list(X.columns)}
    return X, yv, meta


def _build_row(defaults: Dict[str, Any], cat_levels: Dict[str, List[str]], base: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col, default in defaults.items():
        if col in base:
            val = base[col]
            if isinstance(default, (int, float)):
                try:
                    row[col] = float(str(val).replace(",", ".").replace("days", "").strip())
                except Exception:
                    row[col] = float(default)
            else:
                v = str(val).strip().strip('"').strip("'")
                levels = cat_levels.get(col) or []
                row[col] = v if (not levels or v in levels) else "(other)"
        else:
            row[col] = default
    return row


def run_candidate_ranking(*, df: pd.DataFrame, viz_spec: Dict[str, Any], aggregate_ctx: Dict[str, Any], prompt: str = "", scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Candidate ranking (no optimization):
    - choose a candidate dimension D
    - train RF regression for target KPI y
    - for each D=value, predict y under fixed scenario inputs
    - rank by objective direction
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    scenario = scenario or {}
    if not isinstance(scenario, dict):
        scenario = {}

    y = _pick_numeric_target(df, prompt)
    if not y:
        return {"text": "Candidate ranking requires a numeric KPI target (none detected).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    direction = _objective_direction(prompt)
    cand_dim = _pick_candidate_dimension(df, prompt, y)
    if not cand_dim:
        return {"text": f"Could not identify a categorical candidate dimension to rank (target='{y}'). Mention a column like resource_group or avg_type.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    X, yv, meta = _prep_features(df, y, max_levels=20)
    if len(X) < 80 or X.shape[1] < 2:
        return {"text": f"Not enough usable rows/features to model '{y}' for candidate ranking.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    # train/test split
    rng = np.random.RandomState(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = yv.iloc[tr], yv.iloc[te]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))

    # Candidate values
    cand_series = df[cand_dim].astype("string").str.strip()
    vc = cand_series.value_counts(dropna=True)
    # avoid extremely rare candidates
    cand_values = vc[vc >= 5].index.astype(str).tolist()
    if len(cand_values) < 2:
        cand_values = vc.index.astype(str).tolist()

    defaults: Dict[str, Any] = meta["defaults"]
    cat_levels: Dict[str, List[str]] = meta["cat_levels"]

    # Rank each candidate by predicted y
    rows: List[Dict[str, Any]] = []
    for v in cand_values:
        base = dict(scenario)
        base[cand_dim] = v
        raw_row = _build_row(defaults, cat_levels, base)
        raw_df = pd.DataFrame([raw_row])
        ohe = pd.get_dummies(raw_df, drop_first=False).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for c in X.columns:
            if c not in ohe.columns:
                ohe[c] = 0.0
        ohe = ohe[X.columns]
        y_hat = float(model.predict(ohe)[0])
        rows.append({"candidate": v, "predicted": y_hat, "n_obs": int(vc.get(v, 0))})

    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return {"text": f"No candidates found for dimension '{cand_dim}'.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    ascending = True if direction == "minimize" else False if direction == "maximize" else True
    rank_df = rank_df.sort_values("predicted", ascending=ascending).reset_index(drop=True)

    top_k = min(10, len(rank_df))
    top = rank_df.head(top_k).copy()

    # Plot: top candidates (keep readable even if some values are extreme)
    fig = plt.figure(figsize=(11, 4.8))
    ax = fig.add_subplot(111)

    plot = top.copy()
    plot["candidate_label"] = plot.apply(lambda r: f"{r['candidate']} (n={int(r['n_obs'])})", axis=1)
    ax.bar(plot["candidate_label"].astype(str).tolist(), plot["predicted"].astype(float).values)

    local_viz = dict(viz_spec or {})
    local_viz["title"] = local_viz.get("title") or "Candidate ranking"
    local_viz["subtitle"] = local_viz.get("subtitle") or f"Objective={direction} | target={y} | dimension={cand_dim}"
    local_viz.setdefault("x", {})
    local_viz.setdefault("y", {})
    local_viz["x"]["label"] = local_viz["x"].get("label") or f"{cand_dim} values (with n)"
    local_viz["y"]["label"] = local_viz["y"].get("label") or f"Predicted {y}"

    df_for_viz = plot[["candidate_label", "predicted"]].rename(columns={"candidate_label": "candidate", "predicted": "y_hat"})
    apply_viz(ax=ax, df=df_for_viz, viz_spec=local_viz, x="candidate", y="y_hat", group=None)

    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_ha("right")
    figs: List[Any] = [fig]

    scen_pairs = ", ".join([f"{k}={v}" for k, v in scenario.items()]) if scenario else "(none provided)"
    lines: List[str] = []
    lines.append(f"Candidate ranking for KPI '{y}' (RandomForest, no tuning).")
    lines.append(f"- Candidate dimension: **{cand_dim}** (ranking observed values; no optimization).")
    lines.append(f"- Fixed scenario inputs (applied to all candidates): {scen_pairs}")
    lines.append(f"- Model quality (holdout): R²≈{r2:.2f}, MAE≈{mae:.3g}.")
    lines.append("")
    lines.append(f"Top {top_k} candidates (sorted by predicted '{y}' with objective='{direction}'):")
    for i, r in top.iterrows():
        lines.append(f"{i+1}. {r['candidate']} → predicted≈{float(r['predicted']):.3g} (n≈{int(r['n_obs'])})")

    lines.append("")
    lines.append("Notes:")
    lines.append("- Small n per candidate can make ranking unstable; validate top items with deeper diagnostics.")
    lines.append("- This ranks existing candidates only; it does not search for new combinations (no what-if optimization).")

    return {"text": "\n".join(lines), "figures": figs, "x": cand_dim, "y": y, "group": None, "granularity": None}