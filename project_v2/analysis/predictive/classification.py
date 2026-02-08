# analysis/predictive/classification.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


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


def _tufte_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.18)
    ax.grid(False, axis="x")


def _pick_target(df: pd.DataFrame, resolved_y: Optional[str], prompt: str, scenario: Dict[str, Any]) -> Optional[str]:
    p = (prompt or "").lower()
    scenario_cols = set([str(k) for k in (scenario or {}).keys()])

    # 1) prompt-mentioned categorical
    cols_by_length = sorted(list(df.columns), key=lambda x: len(str(x)), reverse=True)
    for c in cols_by_length:
        cl = str(c).lower()
        if cl in p and c not in scenario_cols:
            s = df[c]
            if _is_categorical_like(s) and int(s.nunique(dropna=True)) >= 2:
                return c

    # 2) resolved
    if isinstance(resolved_y, str) and resolved_y in df.columns and resolved_y not in scenario_cols:
        s = df[resolved_y]
        if _is_categorical_like(s) and int(s.nunique(dropna=True)) >= 2:
            return resolved_y

    # 3) fallback 2..10 classes
    for c in df.columns:
        if _is_id_like(c) or c in scenario_cols:
            continue
        s = df[c]
        if _is_categorical_like(s):
            nun = int(s.nunique(dropna=True))
            if 2 <= nun <= 10:
                return c

    return None


def _prep_features(df: pd.DataFrame, y: str, *, max_levels: int = 20) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    d = df.copy()

    yv = d[y].astype("string").str.strip().fillna("(missing)")
    Xraw = d.drop(columns=[y])

    defaults: Dict[str, Any] = {}
    ranges: Dict[str, Any] = {}
    cat_levels: Dict[str, List[str]] = {}

    for c in list(Xraw.columns):
        if _is_id_like(c):
            Xraw = Xraw.drop(columns=[c])
            continue

        if pd.api.types.is_numeric_dtype(Xraw[c]):
            s = pd.to_numeric(Xraw[c], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            defaults[c] = med
            q1 = float(s.quantile(0.01)) if s.notna().any() else np.nan
            q99 = float(s.quantile(0.99)) if s.notna().any() else np.nan
            ranges[c] = {"p01": q1, "p99": q99}
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

    meta = {
        "defaults": defaults,
        "ranges": ranges,              # numeric raw cols
        "cat_levels": cat_levels,      # categorical raw cols
        "max_levels": max_levels,
        "raw_columns": list(Xraw.columns),
        "ohe_columns": list(X.columns),
    }
    return X, yv, meta


def _build_scenario_row(meta: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    defaults: Dict[str, Any] = meta.get("defaults", {})
    ranges: Dict[str, Any] = meta.get("ranges", {})
    cat_levels: Dict[str, List[str]] = meta.get("cat_levels", {})

    row: Dict[str, Any] = {}
    warns: List[str] = []

    for col, default in defaults.items():
        if col in scenario:
            raw_val = scenario[col]

            if col in ranges:
                try:
                    v = float(str(raw_val).replace(",", ".").replace("days", "").strip())
                    row[col] = v
                    q = ranges.get(col) or {}
                    p01, p99 = q.get("p01"), q.get("p99")
                    if np.isfinite(p01) and np.isfinite(p99) and (v < p01 or v > p99):
                        warns.append(f"{col}={v} outside typical range (p01≈{p01:.3g}, p99≈{p99:.3g})")
                except Exception:
                    row[col] = float(default)
                    warns.append(f"Could not parse '{col}={raw_val}', used default≈{default:.3g}")
            else:
                val = str(raw_val).strip().strip('"').strip("'")
                levels = cat_levels.get(col) or []
                if levels and val not in levels:
                    row[col] = "(other)"
                    warns.append(f"{col}='{val}' mapped to '(other)'")
                else:
                    row[col] = val if val else default
        else:
            row[col] = default

    return pd.DataFrame([row]), warns


def _aggregate_importances_to_raw(X_columns: List[str], importances: np.ndarray, meta: Dict[str, Any]) -> pd.DataFrame:
    imp_map = dict(zip(X_columns, importances))
    ranges: Dict[str, Any] = meta.get("ranges", {}) or {}
    cat_levels: Dict[str, Any] = meta.get("cat_levels", {}) or {}

    rows = []

    for raw in ranges.keys():
        if raw in imp_map:
            rows.append({"raw_feature": raw, "importance": float(imp_map[raw])})

    for raw in cat_levels.keys():
        prefix = f"{raw}_"
        s = 0.0
        any_hit = False
        for c, v in imp_map.items():
            if c.startswith(prefix):
                s += float(v)
                any_hit = True
        if any_hit:
            rows.append({"raw_feature": raw, "importance": float(s)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce").fillna(0.0)
    return out.sort_values("importance", ascending=False)


def run_classification(
    *,
    df: pd.DataFrame,
    viz_spec: Dict[str, Any],
    aggregate_ctx: Dict[str, Any],
    prompt: str = "",
    scenario: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compact classification output:
    - Left: confusion matrix (normalized + counts)
    - Right: top feature importances (aggregated to raw features)
    - Bottom row (spanning both plots): comprehensive textbox with model metrics + scenario prediction

    Scenario values are used for inference only (not as filters).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"text": "No data available (empty dataframe).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    scenario = scenario or {}
    if not isinstance(scenario, dict):
        scenario = {}

    resolved = (viz_spec or {}).get("resolved") or {}
    y = _pick_target(df, resolved.get("y"), prompt=prompt, scenario=scenario)
    if not y:
        return {"text": "No suitable categorical target found for classification (need 2–10 stable classes).", "figures": [], "x": None, "y": None, "group": None, "granularity": None}

    X, yv, meta = _prep_features(df, y, max_levels=20)

    # bucket ultra-rare classes
    vc = yv.value_counts(dropna=False)
    keep = set(vc[vc >= 5].index.tolist())
    yv2 = yv.where(yv.isin(keep), "(other)")
    if int(yv2.nunique(dropna=True)) < 2:
        return {"text": f"Target '{y}' has too few stable classes after bucketing rare labels.", "figures": [], "x": None, "y": y, "group": None, "granularity": None}

    # split
    rng = np.random.RandomState(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = yv2.iloc[tr], yv2.iloc[te]

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="weighted"))

    classes = list(model.classes_)
    cm = confusion_matrix(y_test, pred, labels=classes)

    # normalize for business clarity (row-wise: “of actual class, how often predicted as ...”)
    cm_row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.divide(cm, np.where(cm_row_sum == 0, 1.0, cm_row_sum))

    # importances (raw-aggregated)
    imp = getattr(model, "feature_importances_", None)
    imp_raw = pd.DataFrame([])
    if imp is not None and len(imp) == X.shape[1]:
        imp_raw = _aggregate_importances_to_raw(list(X.columns), imp, meta).head(10)

    # scenario inference for textbox
    pred_cls = None
    scen_warns: List[str] = []
    if scenario:
        raw_row, scen_warns = _build_scenario_row(meta, scenario)
        raw_row_ohe = pd.get_dummies(raw_row, drop_first=False).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for c in X.columns:
            if c not in raw_row_ohe.columns:
                raw_row_ohe[c] = 0.0
        raw_row_ohe = raw_row_ohe[X.columns]
        pred_cls = str(model.predict(raw_row_ohe)[0])

    # ---- Plot: 2 panels + dedicated bottom textbox row ----
    fig = plt.figure(figsize=(13.2, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[12.0, 2.4], width_ratios=[1.05, 1.0])

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, :])
    axT.axis("off")

    # Left: confusion matrix
    axL.set_title(f"Confusion matrix (normalized) | target={y}", fontsize=10, pad=10)
    axL.imshow(cm_norm, aspect="auto")
    axL.set_xticks(range(len(classes)))
    axL.set_yticks(range(len(classes)))
    axL.set_xticklabels([str(c) for c in classes], rotation=30, ha="right")
    axL.set_yticklabels([str(c) for c in classes])
    axL.set_xlabel("Predicted")
    axL.set_ylabel("Actual")

    # annotate: % plus count (compact)
    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = cm_norm[i, j] * 100.0
            axL.text(
                j, i,
                f"{pct:.0f}%\n({cm[i, j]})",
                ha="center", va="center",
                fontsize=8,
            )

    # Right: top feature importances (vertical bars, fixed width)
    _tufte_axes(axR)
    if imp_raw.empty:
        axR.set_title("Top feature importances", fontsize=10, pad=10)
        axR.text(0.5, 0.5, "No importances available.", ha="center", va="center", fontsize=10)
        axR.set_xticks([])
        axR.set_yticks([])
    else:
        labels = imp_raw["raw_feature"].astype(str).tolist()
        vals = imp_raw["importance"].astype(float).tolist()
        k = len(vals)
        slots = max(10, k)  # fixed feel
        x = np.arange(slots)
        heights = np.zeros(slots, dtype=float)
        heights[:k] = np.array(vals, dtype=float)

        axR.bar(x, heights, width=0.65)
        axR.set_title(f"Top feature importances (k={k})", fontsize=10, pad=10)
        axR.set_xticks(np.arange(k))
        axR.set_xticklabels(labels, rotation=30, ha="right")
        axR.set_ylabel("Importance")

    # Overall title
    fig.suptitle("Classification", fontsize=12, y=1.02)

    # Bottom textbox: comprehensive (model + prediction)
    model_info = f"Target: {y} | Accuracy≈{acc:.2f} | Weighted F1≈{f1:.2f} | Classes={len(classes)}"
    if scenario:
        scen_pairs = ", ".join([f"{k}={v}" for k, v in scenario.items()])
        pred_info = f"Scenario prediction: {y} = {pred_cls} | Inputs: {scen_pairs}"
        notes = ("Notes: " + " | ".join(scen_warns[:2])) if scen_warns else ""
        box_text = model_info + "\n" + pred_info + (("\n" + notes) if notes else "")
    else:
        box_text = model_info + "\nScenario prediction: (no scenario provided)"

    axT.text(
        0.5, 0.5,
        box_text,
        ha="center", va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", alpha=0.55),
        transform=axT.transAxes,
    )

    text = (
        f"Classification model for target '{y}'. Holdout: Accuracy≈{acc:.2f}, Weighted F1≈{f1:.2f}. "
        f"Scenario values are used for inference only (not as filters)."
    )
    if pred_cls is not None:
        text += f" Scenario prediction: {y}={pred_cls}."

    return {"text": text, "figures": [fig], "x": None, "y": y, "group": None, "granularity": None}