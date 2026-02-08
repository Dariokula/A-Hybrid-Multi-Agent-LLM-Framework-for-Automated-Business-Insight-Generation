# agents/llm_ollama.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import json
from json import JSONDecodeError

import httpx

from agents.llm import LLMProvider, _safe_dumps, _extract_json_object


def _is_context_error(resp_text: str, status_code: int) -> bool:
    t = (resp_text or "").lower()
    if status_code in (400, 413, 422, 500):
        return (
            "context" in t
            or "too long" in t
            or "prompt is too long" in t
            or "maximum context length" in t
            or "token" in t and "limit" in t
        )
    return False


def _shrink_obj(obj: Any, *, max_str: int, max_list: int, max_depth: int, _depth: int = 0) -> Any:
    """Generic recursive compactor: truncates long strings, limits list sizes, limits depth."""
    if _depth >= max_depth:
        # stop recursion; stringify remaining
        try:
            return str(obj)
        except Exception:
            return "<truncated>"

    if obj is None:
        return None

    if isinstance(obj, str):
        return obj if len(obj) <= max_str else (obj[: max_str - 20] + " …<truncated>…")

    if isinstance(obj, (int, float, bool)):
        return obj

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            # keep keys, shrink values
            out[str(k)] = _shrink_obj(v, max_str=max_str, max_list=max_list, max_depth=max_depth, _depth=_depth + 1)
        return out

    if isinstance(obj, (list, tuple)):
        seq = list(obj)
        if len(seq) > max_list:
            seq = seq[:max_list] + [f"<truncated_list tail={len(obj)-max_list}>"]
        return [_shrink_obj(x, max_str=max_str, max_list=max_list, max_depth=max_depth, _depth=_depth + 1) for x in seq]

    # fallback
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _compact_user_payload(user_payload: Dict[str, Any], level: int) -> Dict[str, Any]:
    """
    Level 0: original
    Level 1: shrink strings/lists in df_profile + domain_knowledge snippets
    Level 2: drop df_profile details + keep only tiny DK + minimal meta
    """
    payload = dict(user_payload or {})
    meta = dict(payload.get("meta") or {})
    df_profile = payload.get("df_profile") or {}

    if level == 0:
        return payload

    if level == 1:
        # Keep structure, but shrink heavy text.
        if "domain_knowledge" in meta:
            dk = meta.get("domain_knowledge") or {}
            # shrink snippets aggressively
            dk2 = _shrink_obj(dk, max_str=1200, max_list=3, max_depth=4)
            meta["domain_knowledge"] = dk2

        df_profile2 = _shrink_obj(df_profile, max_str=800, max_list=20, max_depth=5)
        payload["meta"] = meta
        payload["df_profile"] = df_profile2
        payload["_context_compaction_level"] = 1
        return payload

    # level >= 2: go ultra-compact
    # Keep only essential df schema
    dfp_min = {}
    if isinstance(df_profile, dict):
        for k in ("rows", "cols", "dtypes"):
            if k in df_profile:
                dfp_min[k] = df_profile[k]

    # Keep minimal DK: maybe only counts/debug, no long snippets
    if "domain_knowledge" in meta:
        dk = meta.get("domain_knowledge") or {}
        dk_min = {}
        if isinstance(dk, dict):
            # keep only counts/debug-like fields
            for k in ("debug",):
                if k in dk:
                    dk_min[k] = _shrink_obj(dk[k], max_str=400, max_list=5, max_depth=3)
            # if you want *one* snippet title-only, keep it:
            snippets = dk.get("snippets")
            if isinstance(snippets, list) and snippets:
                first = snippets[0]
                if isinstance(first, dict):
                    dk_min["snippets"] = [{"source": first.get("source"), "title": first.get("title")}]
        meta["domain_knowledge"] = dk_min

    # also shrink meta generally (but keep step/family/type/allowed_types)
    meta_keep = {}
    for k in ("step", "family", "type", "allowed_types", "filters", "columns", "prepare_actions"):
        if k in meta:
            meta_keep[k] = meta[k]
    # keep dk too
    if "domain_knowledge" in meta:
        meta_keep["domain_knowledge"] = meta["domain_knowledge"]

    payload["meta"] = _shrink_obj(meta_keep, max_str=600, max_list=10, max_depth=4)
    payload["df_profile"] = _shrink_obj(dfp_min, max_str=600, max_list=50, max_depth=3)
    payload["_context_compaction_level"] = 2
    return payload


class OllamaJSONLLM(LLMProvider):
    """
    Ollama /api/chat wrapper, JSON-only.
    Features:
      - format="json"
      - auto-retry with context compaction when prompt is too large
      - JSON repair pass when output is invalid
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_predict: int = 512,
        num_ctx: int | None = 8192,
        timeout_read_s: float = 900.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)
        self.num_ctx = num_ctx
        self.timeout = httpx.Timeout(connect=10.0, read=timeout_read_s, write=10.0, pool=10.0)

    async def _post_chat(self, req: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=req)
            # don't raise yet; we want body for context errors
            if r.status_code >= 400:
                txt = r.text or ""
                raise httpx.HTTPStatusError(
                    f"HTTP {r.status_code}: {txt[:500]}",
                    request=r.request,
                    response=r,
                )
            return r.json()

    async def _call_once(self, *, instruction: str, user_payload: Dict[str, Any]) -> str:
        payload_txt = _safe_dumps(user_payload)

        options = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = int(self.num_ctx)

        req = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "messages": [
                {"role": "system", "content": (instruction or "").strip()},
                {"role": "user", "content": "json\n" + payload_txt},
            ],
            "options": options,
        }

        data = await self._post_chat(req)
        return ((data.get("message") or {}).get("content") or "").strip()

    async def _repair_json(self, *, bad_jsonish: str) -> Dict[str, Any]:
        # Ask the model to output valid JSON only
        options = {"temperature": 0.0, "num_predict": self.num_predict}
        if self.num_ctx is not None:
            options["num_ctx"] = int(self.num_ctx)

        req = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You fix JSON. Return ONLY valid JSON. "
                        "Do not add commentary. Keep the same keys and intended values."
                    ),
                },
                {"role": "user", "content": bad_jsonish},
            ],
            "options": options,
        }
        data = await self._post_chat(req)
        txt = ((data.get("message") or {}).get("content") or "").strip()
        txt2 = _extract_json_object(txt)
        return json.loads(txt2)

    async def complete_json(self, *, instruction: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        base_instr = (instruction or "").strip()
        if base_instr:
            base_instr += "\n\nReturn ONLY valid JSON (no markdown, no extra text)."
        else:
            base_instr = "Return ONLY valid JSON (no markdown, no extra text)."

        # 1) Try up to 3 times, shrinking context on context errors
        last_txt = ""
        for level in (0, 1, 2):
            payload_try = _compact_user_payload(user_payload, level=level)
            try:
                last_txt = await self._call_once(instruction=base_instr, user_payload=payload_try)
                if not last_txt:
                    continue
                txt2 = _extract_json_object(last_txt)
                try:
                    return json.loads(txt2)
                except JSONDecodeError:
                    # 2) Repair pass
                    return await self._repair_json(bad_jsonish=txt2)
            except httpx.HTTPStatusError as e:
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", 0) if resp is not None else 0
                body = (getattr(resp, "text", "") or "")
                if _is_context_error(body, status):
                    # context too large -> continue with next compaction level
                    continue
                # other errors are real
                raise

        head = (last_txt or "")[:400].replace("\n", "\\n")
        tail = (last_txt or "")[-200:].replace("\n", "\\n")
        raise ValueError(f"Ollama LLM did not return valid JSON after retries. head={head} ... tail={tail}")