from __future__ import annotations

"""Domain Knowledge selection helper.

Goal:
    - Keep the expensive DK search out of every pipeline step.
    - Still allow engine/steps to call this function for each step, but return
      cached results for the same prompt/context.

Notes:
    - This module is used both by the pipeline engine (for step-input previews)
      and by agents (to provide compact DK context to the LLM).
    - We intentionally keep the cache key *step-agnostic* by default. That means
      DK is selected once per prompt/family/type/columns and reused.
"""

from typing import Any, Dict, Optional
import hashlib
import json

from domain_knowledge.store import DomainKnowledgeStore


_STORE: Optional[DomainKnowledgeStore] = None
_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_store() -> DomainKnowledgeStore:
    global _STORE
    if _STORE is None:
        _STORE = DomainKnowledgeStore(root_dir="domain_knowledge")
    return _STORE


def _cache_key(*, prompt: str, fam: str, typ: str, cols_txt: str) -> str:
    payload = {
        "prompt": prompt or "",
        "family": fam or "",
        "type": typ or "",
        "cols": cols_txt or "",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def clear_domain_knowledge_cache() -> None:
    """Clear in-process DK selection cache."""
    _CACHE.clear()


def select_domain_knowledge(
    *,
    step: Optional[str] = None,
    prompt: str,
    params: Optional[Dict[str, Any]] = None,
    df_profile: Optional[Dict[str, Any]] = None,
    # Backwards compatibility: some steps previously called select_domain_knowledge(prompt, state=state)
    state: Any = None,
    top_k: int = 5,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Select relevant domain knowledge snippets.

    The engine currently calls this per step. With caching enabled, we only run
    the expensive store.search() once per prompt/context.

    Args:
        step: step name (optional). Used only to enrich the query string.
        prompt: composed prompt.
        params: pipeline params dict (preferred). If omitted and `state` is
            provided, we use state.params.
        df_profile: dataframe profile (optional).
        state: PipelineState (optional, for backwards compatibility).
        top_k: number of snippets.
        use_cache: whether to cache by (prompt,family,type,cols).
    """

    if params is None:
        if state is not None and hasattr(state, "params"):
            params = getattr(state, "params") or {}
        else:
            params = {}
    if df_profile is None:
        df_profile = {}

    fam = str(params.get("family", "") or "")
    typ = str(params.get("type", "") or "")
    cols = params.get("columns", [])
    if isinstance(cols, list) and cols:
        cols_txt = " ".join([str(c) for c in cols[:8]])
    else:
        # best-effort fallback: use schema column names if available
        schema = (df_profile.get("schema") or {}) if isinstance(df_profile, dict) else {}
        dtypes = schema.get("dtypes") or {}
        if isinstance(dtypes, dict) and dtypes:
            cols_txt = " ".join([str(c) for c in list(dtypes.keys())[:8]])
        else:
            cols_txt = ""

    key = _cache_key(prompt=prompt, fam=fam, typ=typ, cols_txt=cols_txt)
    if use_cache and key in _CACHE:
        out = dict(_CACHE[key])
        dbg = out.get("debug") or {}
        if isinstance(dbg, dict):
            dbg = {**dbg, "cache": "hit"}
            out["debug"] = dbg
        return out

    # Build query. (We include step so search can benefit from it, but we do NOT
    # include step in the cache key so DK is reused across steps.)
    step_txt = str(step or "")
    query = f"{prompt}\nfamily={fam}\ntype={typ}\ncols={cols_txt}\nstep={step_txt}".strip()

    store = _get_store()
    out = store.search(query, top_k=int(top_k))

    if not isinstance(out, dict):
        out = {"snippets": [], "debug": {"selected": 0, "reason": "invalid_store_output"}}
    if "snippets" not in out or not isinstance(out.get("snippets"), list):
        out["snippets"] = []
    dbg = out.get("debug") or {}
    if not isinstance(dbg, dict):
        dbg = {}
    dbg = {
        **dbg,
        "cache": "miss" if use_cache else "off",
        "cache_key": key[:10],
        "top_k": int(top_k),
    }
    out["debug"] = dbg

    if use_cache:
        _CACHE[key] = dict(out)
    return out