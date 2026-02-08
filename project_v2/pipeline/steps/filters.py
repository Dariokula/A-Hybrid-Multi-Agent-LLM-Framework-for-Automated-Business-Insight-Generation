from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import re
import pandas as pd
import numpy as np


_ALLOWED_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "contains"}


def _classify_value(val: Any) -> Tuple[Any, str]:
    if val is None:
        return None, "none"
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return None, "none"
        if len(val) == 1:
            return val[0], "scalar"
        return list(val), "list"
    return val, "scalar"


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_text_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _is_datetime_col(df: pd.DataFrame, col: str) -> bool:
    try:
        return col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])
    except Exception:
        return False


def _apply_one_filter(df: pd.DataFrame, f: Dict[str, Any]) -> pd.DataFrame:
    col = f.get("column")
    op = f.get("op")
    raw_val = f.get("value")

    if not isinstance(col, str) or not isinstance(op, str):
        return df.iloc[0:0]
    if col not in df.columns or op not in _ALLOWED_OPS:
        return df.iloc[0:0]

    s = df[col]
    val, kind = _classify_value(raw_val)

    is_text_col = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    s_txt = _safe_text_series(s) if is_text_col else s

    if op == "==":
        if kind == "list":
            vals = [str(x).strip() for x in val] if is_text_col else val
            return df[s_txt.isin(vals)]
        if kind == "none":
            return df[s.isna()]
        if is_text_col and isinstance(val, str):
            return df[s_txt == val.strip()]
        return df[s == val]

    if op == "!=":
        if kind == "list":
            vals = [str(x).strip() for x in val] if is_text_col else val
            return df[~s_txt.isin(vals)]
        if kind == "none":
            return df[~s.isna()]
        if is_text_col and isinstance(val, str):
            return df[s_txt != val.strip()]
        return df[s != val]

    if op in {">", ">=", "<", "<="}:
        if kind != "scalar" or kind == "none":
            return df.iloc[0:0]
        try:
            vnum = float(val)
        except Exception:
            return df.iloc[0:0]
        sn = _to_numeric_series(s)
        if op == ">":
            return df[sn > vnum]
        if op == ">=":
            return df[sn >= vnum]
        if op == "<":
            return df[sn < vnum]
        if op == "<=":
            return df[sn <= vnum]

    if op == "in":
        if kind == "list":
            vals = [str(x).strip() for x in val] if is_text_col else val
            return df[s_txt.isin(vals)]
        if kind == "scalar":
            v = val.strip() if (is_text_col and isinstance(val, str)) else val
            return df[s_txt.isin([v])]
        return df.iloc[0:0]

    if op == "contains":
        if kind == "list":
            tokens = [str(x) for x in val if x is not None]
            if not tokens:
                return df.iloc[0:0]
            mask = pd.Series(False, index=df.index)
            base = s.astype(str)
            for t in tokens:
                mask = mask | base.str.contains(t, case=False, na=False)
            return df[mask]
        if kind == "none":
            return df.iloc[0:0]
        return df[s.astype(str).str.contains(str(val), case=False, na=False)]

    return df.iloc[0:0]


def _fallback_filters_decision() -> Dict[str, Any]:
    return {
        "filters": [],
        "rationale": "LLM unavailable; no filters applied (safe fallback).",
        "confidence": 0.25,
        "signals": ["llm_failed", "no_filters"],
        "llm_error": None,
    }


def _explicit_filter_command(prompt: str) -> bool:
    """
    Very strict: only treat as explicit training filter request if the user actually says "filter/filtern".
    (We do NOT interpret phrases like "most extreme" as a request to filter the dataset.)
    """
    p = (prompt or "").lower()
    markers = [
        "filter", "filters", "filtered",
        "filtern", "filtere", "gefiltert", "filterung",
        "subset", "restrict to", "limit to",
    ]
    return any(m in p for m in markers)


def _scenario_payload_from_filters(filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    bounds: Dict[str, Dict[str, float]] = {}

    for f in filters:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        op = f.get("op")
        val = f.get("value")
        if not isinstance(col, str) or not isinstance(op, str):
            continue

        if op == "==":
            payload[col] = val
            continue

        if op in {">", ">=", "<", "<="}:
            try:
                v = float(val)
            except Exception:
                continue
            b = bounds.setdefault(col, {})
            if op in {">", ">="}:
                b["lo"] = v
            else:
                b["hi"] = v
            continue

        if op == "in":
            if isinstance(val, (list, tuple)) and len(val) > 0:
                payload[col] = val[0]
            else:
                payload[col] = val

    for col, b in bounds.items():
        lo = b.get("lo")
        hi = b.get("hi")
        if lo is not None and hi is not None:
            payload[col] = (float(lo) + float(hi)) / 2.0
        elif lo is not None:
            payload[col] = float(lo)
        elif hi is not None:
            payload[col] = float(hi)

    return payload


def _is_predictive_scenario_prompt(prompt: str) -> bool:
    p = (prompt or "").lower()

    scenario_markers = [
        "what ", "will be", "would be", "if ", " given ", " when ", " wenn ", " falls ",
        "we have", "wir haben", "unter der bedingung", "bei ",
    ]
    train_subset_markers = [
        "only ", "nur ", "train only", "train on", "using only", "fit only",
        "restrict to", "subset", "filter to", "filtered to",
    ]

    has_scenario = any(m in p for m in scenario_markers) and (
        "predict" in p or "classif" in p or "forecast" in p or "what" in p
    )
    has_train_subset = any(m in p for m in train_subset_markers)

    return bool(has_scenario) and not bool(has_train_subset)


def _find_scenario_split_index(prompt: str) -> Optional[int]:
    p = (prompt or "").lower()
    markers = [" if ", " given ", " when ", " wenn ", " falls ", " unter der bedingung", " bei "]
    idxs = [p.find(m) for m in markers if p.find(m) >= 0]
    if not idxs:
        return None
    return min(idxs)


def _time_candidates(df_profile: Dict[str, Any]) -> set:
    roles = (df_profile or {}).get("roles") or {}
    cands = roles.get("time_candidates") or []
    out = set()
    if isinstance(cands, (list, tuple)):
        for item in cands:
            # profiling.py emits dict entries like {"column": "...", "parse_rate": ...}
            if isinstance(item, dict):
                col = item.get("column")
                if isinstance(col, str) and col.strip():
                    out.add(col.strip())
            elif isinstance(item, str) and item.strip():
                out.add(item.strip())
    return out


def _heuristic_filters_from_prompt(prompt: str, df_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Conservative fallback: infer a simple equality filter from the prompt + df_profile.

    Only triggers when:
    - Prompt clearly mentions a label (e.g., quoted token or 'labeled/label/mit ...')
    - The value exists among top categorical values in df_profile
    """
    p = (prompt or "").strip()
    if not p:
        return []

    p_l = p.lower()

    # 1) Try quoted value first: "mit TA", 'finished', ...
    quoted = re.findall(r"['\"]([^'\"]{1,80})['\"]", p)
    candidates: List[str] = [q.strip() for q in quoted if q and q.strip()]

    # 2) Try common phrasing without quotes: labeled X / label X / mit X
    m = re.search(r"\b(?:labeled|label(?:ed)?|mit)\s+([^\.,;:\n]{1,60})", p_l)
    if m:
        cand = m.group(1).strip()
        # stop at typical tail words
        cand = re.split(r"\b(for|of|in|by|on|and|with)\b", cand)[0].strip()
        if cand:
            candidates.append(cand)

    # de-dup, keep order
    seen = set()
    cands2: List[str] = []
    for c in candidates:
        c0 = c.strip()
        if not c0:
            continue
        key = c0.lower()
        if key in seen:
            continue
        seen.add(key)
        cands2.append(c0)

    if not cands2:
        return []

    categorical = (df_profile or {}).get("categorical") or {}
    if not isinstance(categorical, dict) or not categorical:
        return []

    # build lookup: value(lower) -> (column, original_value)
    matches: List[Dict[str, str]] = []
    for col, top_list in categorical.items():
        if not isinstance(top_list, list):
            continue
        for item in top_list:
            if not isinstance(item, dict):
                continue
            v = item.get("value")
            if v is None:
                continue
            v_str = str(v).strip()
            if not v_str:
                continue
            matches.append({"col": str(col), "v_lower": v_str.lower(), "v": v_str})

    if not matches:
        return []

    # scoring: prefer (a) column name mentioned in prompt, (b) exact value match, (c) shorter columns
    best = None
    best_score = -1.0

    for cand in cands2:
        cand_l = cand.lower().strip()
        for m0 in matches:
            if m0["v_lower"] != cand_l:
                continue
            col = m0["col"]
            score = 1.0
            if col.lower() in p_l:
                score += 2.0
            # if prompt mentions 'status', slight boost for status-ish columns
            if ("status" in p_l or "label" in p_l or "ta" in p_l) and any(k in col.lower() for k in ["status", "label", "ta"]):
                score += 0.5
            score += max(0.0, 0.2 - (len(col) / 200.0))
            if score > best_score:
                best_score = score
                best = {"column": col, "op": "==", "value": m0["v"]}

    return [best] if best else []


async def step_filters(*, df: pd.DataFrame, state, df_profile: Dict[str, Any], agents):
    prompt = state.composed_prompt()
    llm_error: Optional[str] = None

    try:
        out = await agents.run(
            "filters",
            prompt=prompt,
            meta={"step": "filters", "family": state.params.get("family"), "type": state.params.get("type")},
            df_profile=df_profile,
        )
        if not isinstance(out, dict):
            raise ValueError("filters agent returned non-dict output")
        out.setdefault("llm_error", None)
    except Exception as e:
        llm_error = f"{type(e).__name__}: {e}"
        out = _fallback_filters_decision()
        out["llm_error"] = llm_error

    rows_before = len(df)
    df2 = df

    proposed_filters = out.get("filters", [])
    if not isinstance(proposed_filters, list):
        proposed_filters = []

    # If the model produced no usable filters (common for smaller/local LLMs),
    # try a conservative heuristic based on prompt text + df_profile categorical top values.
    if len(proposed_filters) == 0:
        h = _heuristic_filters_from_prompt(prompt, df_profile)
        if h:
            proposed_filters = h
            out["filters"] = h
            out.setdefault("signals", [])
            out["signals"] = list(out["signals"]) + ["heuristic_prompt_value_filter"]
            out["rationale"] = (out.get("rationale", "") + " Heuristic: inferred a simple label filter from prompt.").strip()

    fam = (state.params.get("family") or "").lower().strip()
    typ = (state.params.get("type") or "").lower().strip()

    # ------------------------------------------------------------------
    # POLICY (hard): anomaly_explanation must preserve baseline.
    # - We NEVER allow numeric threshold filters (>,>=,<,<=) for anomaly_explanation,
    #   unless it's a real datetime column (time-window filters).
    # - Business filters like status == 'finished' remain allowed.
    # - We ignore "most extreme" phrasing; only explicit "filter/filtern" counts.
    # ------------------------------------------------------------------
    if fam == "diagnostic" and typ == "anomaly_explanation":
        time_cands = _time_candidates(df_profile)
        explicit_user_filtering = _explicit_filter_command(prompt)

        kept: List[Dict[str, Any]] = []
        dropped_notes: List[str] = []

        for f in proposed_filters:
            if not isinstance(f, dict):
                dropped_notes.append("Dropped non-dict filter.")
                continue

            col = f.get("column")
            op = f.get("op")
            val = f.get("value")

            if not isinstance(col, str) or col not in df2.columns:
                dropped_notes.append(f"Dropped filter (unknown column): {col!r}")
                continue
            if not isinstance(op, str) or op not in _ALLOWED_OPS:
                dropped_notes.append(f"Dropped filter (unsupported op): {col!r} {op!r}")
                continue

            # If it's a numeric threshold op, drop it unless it's a REAL datetime column.
            if op in {">", ">=", "<", "<="}:
                if not (col in time_cands and _is_datetime_col(df2, col)):
                    dropped_notes.append(
                        f"Dropped threshold filter for anomaly_explanation (baseline must remain): {col} {op} {val!r}"
                    )
                    continue

            # Even if user explicitly wants filtering, we STILL block metric thresholds;
            # explicit filtering is typically for business scope (status/resource_group/etc.).
            if explicit_user_filtering and op in {">", ">=", "<", "<="} and not _is_datetime_col(df2, col):
                dropped_notes.append(
                    f"Dropped metric threshold even under explicit filtering (use outlier mode instead): {col} {op} {val!r}"
                )
                continue

            kept.append({"column": col, "op": op, "value": val})

        proposed_filters = kept
        out["filters"] = kept
        out.setdefault("signals", [])
        out["signals"] = list(out["signals"]) + ["policy_anomaly_explanation_preserve_baseline"]
        out["_policy_dropped_notes"] = dropped_notes

    # ---------------------------
    # Scenario mode (predictive + prescriptive) (unchanged)
    # ---------------------------
    scenario_mode = False
    if fam == "predictive" and typ in {"regression", "classification"} and _is_predictive_scenario_prompt(prompt):
        scenario_mode = True
    if fam == "prescriptive" and typ in {"scenario_evaluation", "candidate_ranking"}:
        scenario_mode = _find_scenario_split_index(prompt) is not None

    scenario_payload: Dict[str, Any] = {}

    if scenario_mode:
        split_idx = _find_scenario_split_index(prompt)
        pre_clause = (prompt or "")[:split_idx].lower() if split_idx is not None else ""
        post_clause = (prompt or "")[split_idx:].lower() if split_idx is not None else (prompt or "").lower()

        explicit_user_filtering = _explicit_filter_command(prompt)

        # predictive: treat constraints as scenario inputs by default
        if fam == "predictive" and typ in {"regression", "classification"} and not explicit_user_filtering:
            dropped_notes: List[str] = []
            scenario_filters: List[Dict[str, Any]] = []

            for f in proposed_filters:
                if not isinstance(f, dict):
                    dropped_notes.append("Dropped non-dict filter.")
                    continue
                col = f.get("column")
                op = f.get("op")
                val = f.get("value")

                if not isinstance(col, str) or col not in df2.columns:
                    dropped_notes.append(f"Dropped filter (unknown column): {col!r}")
                    continue
                if not isinstance(op, str) or op not in _ALLOWED_OPS:
                    dropped_notes.append(f"Dropped filter (unsupported op): {col!r} {op!r}")
                    continue

                scenario_filters.append({"column": col, "op": op, "value": val})

            scenario_payload = _scenario_payload_from_filters(scenario_filters)

            df2 = df
            out["filters"] = []
            out.setdefault("signals", [])
            out["signals"] = list(out["signals"]) + ["scenario_mode", "scenario_inputs_captured", "no_training_filters_unless_explicit"]
            out["rationale"] = (
                (out.get("rationale", "") + " Scenario mode (ML): kept full dataset for training; captured scenario inputs for inference.").strip()
            )

            state.params["scenario"] = scenario_payload
            state.params["scenario_filters"] = scenario_filters
            state.params["filters"] = []

            state.decisions["filters"] = {
                **out,
                "proposed_filters": proposed_filters,
                "kept_filters": [],
                "dropped_notes": dropped_notes,
                "scenario_mode": True,
                "scenario": scenario_payload,
                "scenario_filters": scenario_filters,
                "explicit_training_filter": False,
            }

            meta = {
                "decision": state.decisions["filters"],
                "rationale": out.get("rationale", ""),
                "df_delta": {"rows_before": rows_before, "rows_after": len(df2)},
            }
            if llm_error:
                meta.setdefault("warnings", []).append(f"filters_llm_error: {llm_error}")
            if dropped_notes:
                meta.setdefault("warnings", []).extend(dropped_notes)

            return meta, df2

        kept_filters: List[Dict[str, Any]] = []
        dropped_notes: List[str] = []
        scenario_filters: List[Dict[str, Any]] = []

        df_scope = df2
        for f in proposed_filters:
            if not isinstance(f, dict):
                dropped_notes.append("Dropped non-dict filter.")
                continue

            col = f.get("column")
            op = f.get("op")
            val = f.get("value")

            if not isinstance(col, str) or col not in df_scope.columns:
                dropped_notes.append(f"Dropped filter (unknown column): {col!r}")
                continue
            if not isinstance(op, str) or op not in _ALLOWED_OPS:
                dropped_notes.append(f"Dropped filter (unsupported op): {col!r} {op!r}")
                continue

            col_l = col.lower()
            in_pre = (col_l in pre_clause) if pre_clause else False
            in_post = (col_l in post_clause) if post_clause else True

            if in_post and not in_pre:
                scenario_filters.append({"column": col, "op": op, "value": val})
                if op == "==" and isinstance(col, str):
                    scenario_payload[col] = val
                continue

            df_test = _apply_one_filter(df_scope, f)
            if len(df_test) == 0:
                dropped_notes.append(f"Dropped scope filter {col} {op} {val!r} (0 rows, exact test).")
                continue

            kept_filters.append({"column": col, "op": op, "value": val})
            df_scope = df_test

        df2 = df_scope

        state.params["scenario"] = scenario_payload
        state.params["scenario_filters"] = scenario_filters

        out["filters"] = kept_filters
        out.setdefault("signals", [])
        out["signals"] = list(out["signals"]) + ["scenario_mode", "scenario_inputs_captured"]
        out["rationale"] = (
            (out.get("rationale", "") + " Scenario mode: kept scope filters (pre-clause) and moved scenario constraints (post-clause) to model inference.").strip()
        )

        state.params["filters"] = kept_filters
        state.decisions["filters"] = {
            **out,
            "proposed_filters": proposed_filters,
            "kept_filters": kept_filters,
            "dropped_notes": dropped_notes,
            "scenario_mode": True,
            "scenario": scenario_payload,
            "scenario_filters": scenario_filters,
        }

        meta = {
            "decision": state.decisions["filters"],
            "rationale": out.get("rationale", ""),
            "df_delta": {"rows_before": rows_before, "rows_after": len(df2)},
        }
        if llm_error:
            meta.setdefault("warnings", []).append(f"filters_llm_error: {llm_error}")
        if dropped_notes:
            meta.setdefault("warnings", []).extend(dropped_notes)

        return meta, df2

    # ---------------------------
    # Normal filter application
    # ---------------------------
    kept_filters: List[Dict[str, Any]] = []
    dropped_notes: List[str] = []

    policy_drops = out.pop("_policy_dropped_notes", None)
    if isinstance(policy_drops, list):
        dropped_notes.extend([str(x) for x in policy_drops if x is not None])

    for f in proposed_filters:
        if not isinstance(f, dict):
            dropped_notes.append("Dropped non-dict filter.")
            continue

        col = f.get("column")
        op = f.get("op")
        val = f.get("value")

        if not isinstance(col, str) or col not in df2.columns:
            dropped_notes.append(f"Dropped filter (unknown column): {col!r}")
            continue
        if not isinstance(op, str) or op not in _ALLOWED_OPS:
            dropped_notes.append(f"Dropped filter (unsupported op): {col!r} {op!r}")
            continue

        df_test = _apply_one_filter(df2, f)
        if len(df_test) == 0:
            dropped_notes.append(f"Dropped filter {col} {op} {val!r} (0 rows, exact test).")
            continue

        kept_filters.append({"column": col, "op": op, "value": val})
        df2 = df_test

    if len(kept_filters) == 0 and len(proposed_filters) > 0:
        df2 = df
        out["filters"] = []
        out.setdefault("signals", [])
        out["signals"] = list(out["signals"]) + ["no_filters_after_exact_test"]
        out["rationale"] = (out.get("rationale", "") + " No filter applied (exact test produced 0 matches).").strip()

    state.params.pop("scenario", None)
    state.params.pop("scenario_filters", None)

    state.params["filters"] = out.get("filters", [])
    state.decisions["filters"] = {
        **out,
        "proposed_filters": proposed_filters,
        "kept_filters": kept_filters,
        "dropped_notes": dropped_notes,
        "scenario_mode": False,
    }

    meta = {
        "decision": state.decisions["filters"],
        "rationale": out.get("rationale", ""),
        "df_delta": {"rows_before": rows_before, "rows_after": len(df2)},
    }
    if llm_error:
        meta.setdefault("warnings", []).append(f"filters_llm_error: {llm_error}")
    if dropped_notes:
        meta.setdefault("warnings", []).extend(dropped_notes)

    return meta, df2