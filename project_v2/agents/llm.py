# agents/llm.py
from __future__ import annotations

from typing import Any, Dict, Protocol
import json
import re
from datetime import datetime, date

from openai import AsyncOpenAI


class LLMProvider(Protocol):
    async def complete_json(self, *, instruction: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        ...
        

def _json_default(o: Any):
    """Best-effort JSON serializer for common pandas/numpy/datetime objects."""
    if isinstance(o, (datetime, date)):
        return o.isoformat()

    try:
        import pandas as pd  # type: ignore
        if isinstance(o, pd.Timestamp):
            try:
                return o.isoformat()
            except Exception:
                return str(o)
        if isinstance(o, pd.Timedelta):
            return str(o)
        if o is pd.NaT:
            return None
    except Exception:
        pass

    try:
        import numpy as np  # type: ignore
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
    except Exception:
        pass

    if isinstance(o, (set, tuple)):
        return list(o)

    if isinstance(o, bytes):
        return o.decode("utf-8", errors="replace")

    return str(o)


def _safe_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def _extract_json_object(txt: str) -> str:
    """
    Recover JSON object from model output:
    - strips ```json fences
    - extracts substring from first '{' to last '}'
    """
    if not txt:
        return txt
    s = txt.strip()

    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()

    return s


def _response_text(resp: Any) -> str:
    """Robustly extract text from Responses API response."""
    if resp is None:
        return ""

    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    out = getattr(resp, "output", None)
    if not isinstance(out, list):
        return ""

    parts: list[str] = []
    for item in out:
        content = getattr(item, "content", None)
        if isinstance(content, list):
            for c in content:
                ctype = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")
                if ctype in ("output_text", "text"):
                    txt = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())

        itype = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if itype == "reasoning":
            summary = getattr(item, "summary", None) if not isinstance(item, dict) else item.get("summary")
            if isinstance(summary, list):
                for s in summary:
                    st = getattr(s, "text", None) if not isinstance(s, dict) else s.get("text")
                    if isinstance(st, str) and st.strip():
                        parts.append(st.strip())

    return "\n".join(parts).strip()


class OpenAIJSONLLM:
    """
    JSON-only wrapper.
    Compatible with AgentFactory(): OpenAIJSONLLM(model=..., max_output_tokens=...)
    Adds:
      - safer token floor (finalize often needs >2000)
      - auto-repair retry if output is invalid/truncated JSON
    """

    def __init__(self, *, model: str, max_output_tokens: int = 2000):
        self.client = AsyncOpenAI()
        self.model = model

        # IMPORTANT: finalize JSON can easily exceed 2000 tokens.
        # Set a sensible floor so you don't get truncated JSON strings.
        mot = int(max_output_tokens)
        self.max_output_tokens = max(mot, 4096)

    async def _call_responses_json(self, *, instruction: str, payload_txt: str, max_out: int) -> str:
        # Responses API requires "json" in INPUT when using json_object format.
        responses_input = "json\n" + payload_txt

        resp = await self.client.responses.create(
            model=self.model,
            instructions=instruction,
            input=responses_input,
            max_output_tokens=max_out,
            text={"format": {"type": "json_object"}},
        )
        return _response_text(resp)

    async def _call_chat_json(self, *, instruction: str, payload_txt: str, max_out: int) -> str:
        cc = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": instruction + "\n\njson\n" + payload_txt}],
            response_format={"type": "json_object"},
            # IMPORTANT: newer models reject max_tokens
            max_completion_tokens=max_out,
        )
        return (cc.choices[0].message.content or "").strip()

    async def _repair_json(self, *, original_instruction: str, bad_txt: str, payload_txt: str) -> Dict[str, Any]:
        """
        If model returns truncated/invalid JSON, ask it to output a shorter valid JSON object.
        We include:
          - original instruction (schema)
          - the bad output (for context)
          - the original payload (so it can regenerate)
        """
        repair_instr = (
            (original_instruction or "").strip()
            + "\n\nYour previous output was INVALID or TRUNCATED JSON."
            + "\nReturn ONLY a VALID JSON object that matches the schema."
            + "\nBe concise: keep strings short, limit highlights/caveats/followups."
            + "\nNo markdown, no extra text."
        ).strip()

        # keep the bad output bounded so we don't explode context
        bad_tail = (bad_txt or "")[-2000:]
        repair_payload = _safe_dumps(
            {
                "note": "repair_json",
                "bad_output_tail": bad_tail,
                "original_payload": json.loads(payload_txt) if payload_txt.startswith("{") else payload_txt,
            }
        )
        # try responses first, then chat
        txt = await self._call_responses_json(instruction=repair_instr, payload_txt=repair_payload, max_out=self.max_output_tokens)
        if not txt:
            txt = await self._call_chat_json(instruction=repair_instr, payload_txt=repair_payload, max_out=self.max_output_tokens)

        if not txt:
            raise ValueError("LLM repair attempt returned empty output.")

        txt2 = _extract_json_object(txt)
        return json.loads(txt2)

    async def complete_json(self, *, instruction: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        payload_txt = _safe_dumps(user_payload)

        base_instr = (instruction or "").strip()
        base_instr = (
            base_instr + "\n\nReturn ONLY valid JSON (no markdown, no extra text)."
            if base_instr
            else "Return ONLY valid JSON (no markdown, no extra text)."
        )

        # 1) Responses API
        txt = await self._call_responses_json(
            instruction=base_instr,
            payload_txt=payload_txt,
            max_out=self.max_output_tokens,
        )

        # 2) Fallback: Chat Completions
        if not txt:
            txt = await self._call_chat_json(
                instruction=base_instr,
                payload_txt=payload_txt,
                max_out=self.max_output_tokens,
            )

        if not txt:
            raise ValueError("LLM returned empty output (no output_text / no extractable content).")

        txt2 = _extract_json_object(txt)

        try:
            return json.loads(txt2)
        except Exception:
            # 3) Auto-repair: regenerate concise valid JSON (fixes truncated strings)
            try:
                return await self._repair_json(
                    original_instruction=base_instr,
                    bad_txt=txt,
                    payload_txt=payload_txt,
                )
            except Exception as e2:
                head = txt[:400].replace("\n", "\\n")
                tail = txt[-200:].replace("\n", "\\n")
                raise ValueError(f"LLM did not return valid JSON. head={head} ... tail={tail}") from e2