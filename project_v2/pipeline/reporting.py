# pipeline/reporting.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


# -----------------------------
# Helpers
# -----------------------------
def _get_last_step(results: List[Dict[str, Any]], step_name: str) -> Optional[Dict[str, Any]]:
    for r in reversed(results):
        if r.get("step") == step_name:
            return r
    return None


def _safe_get(d: Any, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# -----------------------------
# Human-friendly prepare action names
# -----------------------------
_PREP_ACTION_LABELS = {
    "coerce_numeric": "Convert to numbers",
    "coerce_datetime": "Convert to dates/times",
    "parse_datetimes": "Convert to dates/times",
    "parse_datetime": "Convert to dates/times",
    "drop_rows_missing_required": "Remove rows with missing required values",
    "dropna": "Remove rows with missing values",
    "fillna": "Fill missing values",
    "impute_numeric_median": "Fill missing numbers (typical value)",
    "impute_categorical_mode": "Fill missing categories (most common)",
    "strip_strings": "Clean up text (trim spaces)",
    "normalize_missing_tokens": "Standardize missing-value labels",
    "drop_duplicates": "Remove duplicates",
    "deduplicate": "Remove duplicates",
    "clip_outliers_iqr": "Cap extreme values",
    "clip": "Cap extreme values",
    "winsorize": "Cap extreme values",
}


def _fmt_cols(cols: Any, max_cols: int = 3) -> str:
    if not isinstance(cols, list) or not cols:
        return ""
    shown = [str(c) for c in cols[:max_cols]]
    extra = len(cols) - len(shown)
    if extra > 0:
        return f"{', '.join(shown)} (+{extra})"
    return ", ".join(shown)


def _humanize_prepare_actions(prep_dec: Any, max_items: int = 4) -> str:
    if not isinstance(prep_dec, dict):
        return "none"

    acts = prep_dec.get("actions") or prep_dec.get("prepare_actions") or []
    if not isinstance(acts, list) or not acts:
        return "none"

    out: List[str] = []
    for a in acts:
        if not isinstance(a, dict):
            continue
        name = (a.get("name") or "").strip()
        params = a.get("params") if isinstance(a.get("params"), dict) else {}
        if not name:
            continue

        label = _PREP_ACTION_LABELS.get(name, name.replace("_", " ").strip().capitalize())

        col_info = ""
        if "column" in params and params.get("column") is not None:
            col_info = str(params.get("column"))
        elif "columns" in params:
            col_info = _fmt_cols(params.get("columns"))
        elif "subset" in params:
            col_info = _fmt_cols(params.get("subset"))

        out.append(f"{label} ({col_info})" if col_info else label)

    if not out:
        return "none"

    shown = out[:max_items]
    s = "; ".join(shown)
    if len(out) > max_items:
        s += f"; … (+{len(out) - max_items})"
    return s


# -----------------------------
# Overview summarizers
# -----------------------------
def _summarize_family_or_type(dec: Any, key: str) -> Optional[str]:
    if not isinstance(dec, dict):
        return None
    val = dec.get(key)
    if not val:
        return None
    conf = _as_float(dec.get("confidence"))
    if conf is None:
        return str(val)
    return f"{val} (confidence={conf:.2f})"


def _summarize_filters(filters_dec: Any) -> str:
    if not isinstance(filters_dec, dict):
        return ""
    flt = filters_dec.get("filters") or []
    if not isinstance(flt, list) or not flt:
        return ""
    parts = []
    for f in flt[:6]:
        if isinstance(f, dict):
            col = f.get("column")
            op = f.get("op")
            val = f.get("value")
            parts.append(f"{col} {op} {val}")
    s = "; ".join(parts)
    if len(flt) > 6:
        s += f"; … (+{len(flt)-6})"
    return s


def _summarize_columns(cols_dec: Any) -> str:
    if not isinstance(cols_dec, dict):
        return ""
    cols = cols_dec.get("columns") or []
    if not isinstance(cols, list) or not cols:
        return ""
    s = ", ".join(map(str, cols[:12]))
    if len(cols) > 12:
        s += f", … (+{len(cols)-12})"
    return s


def _summarize_aggregate(agg_dec: Any) -> str:
    """
    Summarize ANY aggregation (time-based or group-only),
    and explicitly reports when aggregation is disabled.
    """
    if not isinstance(agg_dec, dict):
        return "none"

    plan_needed = agg_dec.get("plan_needed")
    plan = agg_dec.get("plan") or {}

    if plan_needed is False:
        return "none (disabled)"

    if not isinstance(plan, dict) or not plan:
        return "none"

    tc = plan.get("time_column")
    gran = plan.get("time_granularity")
    gb = plan.get("groupby_columns") or []
    mets = plan.get("metrics") or []

    gb_s = _fmt_cols(gb, max_cols=4) if isinstance(gb, list) else ""
    met_parts = []
    if isinstance(mets, list):
        for m in mets[:4]:
            if isinstance(m, dict):
                nm = m.get("name") or ""
                agg = m.get("agg") or ""
                col = m.get("column")
                if col is None and str(agg).lower() == "count":
                    met_parts.append(f"{nm or 'n_records'}=count")
                else:
                    met_parts.append(f"{nm or ''}({agg}:{col})".strip())
    met_s = ", ".join(met_parts) if met_parts else ""

    bits = []
    if tc or gran:
        bits.append(f"time={tc}/{gran}")
    if gb_s:
        bits.append(f"groupby=[{gb_s}]")
    if met_s:
        bits.append(f"metrics=[{met_s}]")

    return "; ".join(bits) if bits else "none"


# -----------------------------
# Run overview (experiment-friendly)
# -----------------------------
def render_run_overview(results: List[Dict[str, Any]]):
    if not results:
        return

    prompt = None
    for r in results:
        si = r.get("step_input") or {}
        if isinstance(si, dict) and si.get("prompt"):
            prompt = si.get("prompt")
            break

    family_step = _get_last_step(results, "family") or {}
    type_step = _get_last_step(results, "type") or {}
    filters_step = _get_last_step(results, "filters") or {}
    columns_step = _get_last_step(results, "columns") or {}
    prepare_step = _get_last_step(results, "prepare") or {}
    aggregate_step = _get_last_step(results, "aggregate") or {}
    verify_step = _get_last_step(results, "verify") or {}

    fam_dec = family_step.get("decision")
    typ_dec = type_step.get("decision")
    filt_dec = filters_step.get("decision")
    col_dec = columns_step.get("decision")
    prep_dec = prepare_step.get("decision")
    agg_dec = aggregate_step.get("decision")
    ver_dec = verify_step.get("decision")

    print("\n--- Run overview (inputs & key decisions) ---")
    if prompt:
        print(f"Prompt: {prompt}")

    fam_s = _summarize_family_or_type(fam_dec, "family")
    if fam_s:
        print(f"Family: {fam_s}")

    typ_s = _summarize_family_or_type(typ_dec, "type")
    if typ_s:
        print(f"Type: {typ_s}")

    s_f = _summarize_filters(filt_dec)
    if s_f:
        print(f"Filters: {s_f}")

    s_c = _summarize_columns(col_dec)
    if s_c:
        print(f"Columns: {s_c}")

    print(f"Prepare actions: {_humanize_prepare_actions(prep_dec)}")
    print(f"Aggregate: {_summarize_aggregate(agg_dec)}")

    if isinstance(ver_dec, dict):
        status = ver_dec.get("status")
        conf = ver_dec.get("confidence")
        if status is not None or conf is not None:
            print(f"Verify status: {status} (confidence={conf})")


# -----------------------------
# Plot rendering
# -----------------------------
def render_analyze_plots(results: List[Dict[str, Any]]):
    def _display_figs(figs: List[Any]):
        if not figs:
            return
        print("\n--- Plots ---")
        seen = set()
        for fig in figs:
            if fig is None:
                continue
            fid = id(fig)
            if fid in seen:
                continue
            seen.add(fid)
            display(fig)
            try:
                plt.close(fig)
            except Exception:
                pass

    analyze = _get_last_step(results, "analyze")
    if analyze:
        arts = analyze.get("artifacts") or {}
        figs = arts.get("figures") or []
        if isinstance(figs, list) and figs:
            _display_figs(figs)
            return
        fig1 = arts.get("figure")
        if fig1 is not None:
            _display_figs([fig1])
            return

    viz = _get_last_step(results, "viz")
    if viz:
        arts = viz.get("artifacts") or {}
        figs = arts.get("figures") or []
        if isinstance(figs, list) and figs:
            _display_figs(figs)
            return
        fig1 = arts.get("figure")
        if fig1 is not None:
            _display_figs([fig1])
            return


# -----------------------------
# Final output: ONE single text, no duplicates warning, no repetition
# -----------------------------
_DUPLICATE_WARNING_PATTERNS = [
    r"\bduplicate\b", r"\bduplicates\b", r"\brepeated\b", r"\breplication\b"
]

_GENERIC_FOLLOWUP_PATTERNS = [
    r"\binvestigate\b",
    r"\blook into\b",
    r"\bcheck\b",
]

_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","as","by","at","from","into","over",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "most","some","few","many","typically","usually","often",
}


def _matches_any(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)


def _normalize_for_dedupe(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[-+]?\d+(\.\d+)?", "#", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _token_set(text: str) -> set[str]:
    t = (text or "").lower()
    t = re.sub(r"[-+]?\d+(\.\d+)?", "#", t)
    toks = set(re.findall(r"[a-zA-Z#]+", t))
    return {x for x in toks if x and x not in _STOPWORDS and len(x) > 2}


def _too_similar(a: str, b: str, thr: float = 0.78) -> bool:
    na = _normalize_for_dedupe(a)
    nb = _normalize_for_dedupe(b)

    # fast substring checks
    if na and nb and (na in nb or nb in na):
        return True

    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return False
    inter = len(sa & sb)
    union = len(sa | sb)
    j = inter / max(1, union)
    return j >= thr


def _pick_best_sentences(narrative: Dict[str, Any]) -> Tuple[str, List[str], Optional[str]]:
    summary = str((narrative.get("summary") or "")).strip()

    highlights = narrative.get("highlights") or []
    if not isinstance(highlights, list):
        highlights = []

    warnings = narrative.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = []

    followups = narrative.get("followups") or []
    if not isinstance(followups, list):
        followups = []

    warnings = [w for w in warnings if not _matches_any(str(w), _DUPLICATE_WARNING_PATTERNS)]

    def score(h: str) -> int:
        t = h.lower()
        sc = 0
        if "most" in t or "typical" in t or "between" in t:
            sc += 30
        if "extreme" in t or "up to" in t or "range" in t or "from" in t:
            sc += 20
        if "outlier" in t:
            sc += 25
        if "average" in t or "mean" in t or "median" in t:
            sc += 5
        return sc

    hl = [str(x).strip() for x in highlights if str(x).strip()]
    hl_sorted = sorted(hl, key=score, reverse=True)

    chosen: List[str] = []
    seen_norm = set()
    if summary:
        seen_norm.add(_normalize_for_dedupe(summary))

    # Choose up to 2 highlights, but avoid near-duplicates
    for h in hl_sorted:
        if not h:
            continue
        nh = _normalize_for_dedupe(h)
        if nh in seen_norm:
            continue
        if summary and _too_similar(summary, h):
            continue
        if any(_too_similar(prev, h) for prev in chosen):
            continue

        chosen.append(h)
        seen_norm.add(nh)
        if len(chosen) >= 2:
            break

    warning_1 = None
    if warnings:
        cand = str(warnings[0]).strip()
        if cand and summary and not _too_similar(summary, cand) and not any(_too_similar(x, cand) for x in chosen):
            warning_1 = cand

    followup_1 = None
    for f in followups:
        fs = str(f).strip()
        if not fs:
            continue
        if _matches_any(fs, _GENERIC_FOLLOWUP_PATTERNS) and len(fs.split()) <= 6:
            continue
        if summary and _too_similar(summary, fs):
            continue
        if any(_too_similar(x, fs) for x in chosen):
            continue
        followup_1 = fs
        break

    if followup_1 is None and followups:
        fs = str(followups[0]).strip()
        if fs and not (summary and _too_similar(summary, fs)) and not any(_too_similar(x, fs) for x in chosen):
            followup_1 = fs

    return summary, chosen, warning_1 or followup_1


def _compact_one_text(narrative: Dict[str, Any]) -> str:
    summary, extras, last_item = _pick_best_sentences(narrative)

    parts: List[str] = []
    if summary:
        parts.append(summary.rstrip(".") + ".")

    for e in extras:
        parts.append(str(e).rstrip(".") + ".")

    if last_item:
        parts.append(str(last_item).rstrip(".") + ".")

    # final dedupe pass over whole assembled text (very small)
    out_parts: List[str] = []
    for p in parts:
        if not p.strip():
            continue
        if any(_too_similar(p, q) for q in out_parts):
            continue
        out_parts.append(p.strip())

    text = " ".join(out_parts)
    return " ".join(text.split())


def render_finalize_one_text(results: List[Dict[str, Any]]):
    fin = _get_last_step(results, "finalize")
    if not fin:
        return
    dec = fin.get("decision") or {}
    narrative = dec.get("narrative") if isinstance(dec, dict) else None
    if not isinstance(narrative, dict):
        return

    print("\n--- Summary ---")
    print(_compact_one_text(narrative))


# -----------------------------
# Main report
# -----------------------------
def show_run_report(
    results: List[Dict[str, Any]],
    *,
    show_head_df=None,
    show_step_inputs: bool = False,
    verbose_steps: bool = True,
    render_final: bool = False,
    final_head_rows: int = 8,
):
    if not results:
        print("No results.")
        return

    if verbose_steps:
        for r in results:
            step = r.get("step", "?")
            print(f"\n=== Step: {step} ===")

            step_input = r.get("step_input") or {}
            if show_step_inputs and isinstance(step_input, dict) and step_input:
                print("\n[1] Step Inputs")
                if step_input.get("prompt"):
                    print(f"  Prompt: {step_input.get('prompt')}")
                if step_input.get("chosen_params_so_far") is not None:
                    print(f"  Chosen params so far: {step_input.get('chosen_params_so_far')}")
                schema = _safe_get(step_input, "df_profile", "schema", default=None)
                if schema:
                    print(f"  DataFrame profile schema: {schema}")

            decision = r.get("decision", {}) if isinstance(r.get("decision"), dict) else r.get("decision")
            rationale = r.get("rationale", "")

            print("\n[2] Step Decision")
            print(f"  Decision: {decision}")
            if rationale:
                print(f"  Rationale: {rationale}")

            print("\n[3] Deterministic Execution / DF Changes")
            df_delta = r.get("df_delta", None)
            print(f"  DF delta: {df_delta if df_delta else '(none)'}")

    if show_head_df is not None:
        print("\n--- Output DF head() ---")
        display(show_head_df.head(final_head_rows))

    if render_final:
        render_run_overview(results)
        render_analyze_plots(results)
        render_finalize_one_text(results)