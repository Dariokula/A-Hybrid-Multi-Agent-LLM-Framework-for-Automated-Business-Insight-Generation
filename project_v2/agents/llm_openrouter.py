# agents/llm_openrouter.py
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import asyncio
import re

from openai import AsyncOpenAI

from agents.llm import LLMProvider, _safe_dumps, _extract_json_object


def _compact_tail(txt: str, n: int = 2000) -> str:
    t = (txt or "").strip()
    if len(t) <= n:
        return t
    return t[-n:]


def _strip_think_blocks(txt: str) -> str:
    """
    Many reasoning models emit <think>...</think> blocks.
    Strip them defensively before JSON extraction.
    """
    if not txt:
        return txt
    return re.sub(r"(?is)<think>.*?</think>", "", txt).strip()


def _loads_first_json_object(txt: str) -> Dict[str, Any]:
    """
    Parse the FIRST JSON object from a string.
    Handles 'Extra data' by using raw_decode.
    """
    s = (txt or "").strip()
    if not s:
        raise ValueError("Empty JSON text")

    dec = json.JSONDecoder()

    # Find first '{'
    i = s.find("{")
    if i < 0:
        raise ValueError("No '{' found in model output")

    obj, end = dec.raw_decode(s, i)
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON is not an object")
    return obj


class OpenRouterJSONLLM(LLMProvider):
    """
    OpenRouter JSON-only wrapper (OpenAI-compatible endpoint).

    Key points:
    - Uses base_url="https://openrouter.ai/api/v1"
    - Enforces JSON mode via response_format={"type":"json_object"} when supported
    - Retries on empty output
    - Repairs invalid/truncated JSON by asking for a shorter valid JSON object
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        max_output_tokens: int = 4096,
        base_url: str = "https://openrouter.ai/api/v1",
        app_url: Optional[str] = None,
        app_name: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 120.0,
    ):
        self.model = model
        self.max_output_tokens = max(512, int(max_output_tokens))

        headers = dict(extra_headers or {})
        # Recommended by OpenRouter (optional but helpful for tracking)
        if app_url:
            headers.setdefault("HTTP-Referer", app_url)
        if app_name:
            headers.setdefault("X-Title", app_name)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers if headers else None,
            timeout=timeout_s,
        )

    async def _call_chat(
        self,
        *,
        instruction: str,
        payload_txt: str,
        max_out: int,
        force_json: bool = True,
    ) -> str:
        """
        Calls OpenRouter /chat/completions in OpenAI-compatible style.
        If force_json=True, requests JSON mode (response_format=json_object).
        """
        messages = [
            {"role": "system", "content": instruction.strip()},
            {"role": "user", "content": "json\n" + payload_txt},
        ]

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
        )

        # Some providers/models support it; if they don't, we'll catch and retry w/o it.
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        # Different clients/providers accept different token params; try both.
        try:
            kwargs["max_tokens"] = max_out
            resp = await self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_out
            resp = await self.client.chat.completions.create(**kwargs)

        txt = (resp.choices[0].message.content or "").strip() if resp and resp.choices else ""
        return txt

    async def _repair_json(self, *, original_instruction: str, bad_txt: str, payload_txt: str) -> Dict[str, Any]:
        repair_instr = (
            (original_instruction or "").strip()
            + "\n\nYour previous output was INVALID or TRUNCATED JSON."
            + "\nReturn ONLY a VALID JSON object that matches the schema."
            + "\nBe concise: keep strings short, limit lists."
            + "\nNo markdown, no extra text."
        ).strip()

        repair_payload = _safe_dumps(
            {
                "note": "repair_json",
                "bad_output_tail": _compact_tail(bad_txt, 2000),
                "original_payload": json.loads(payload_txt) if payload_txt.startswith("{") else payload_txt,
            }
        )

        # Try with JSON-mode first, then without JSON-mode
        txt = await self._call_chat(
            instruction=repair_instr,
            payload_txt=repair_payload,
            max_out=self.max_output_tokens,
            force_json=True,
        )
        if not txt:
            txt = await self._call_chat(
                instruction=repair_instr,
                payload_txt=repair_payload,
                max_out=self.max_output_tokens,
                force_json=False,
            )

        if not txt:
            raise ValueError("OpenRouter repair attempt returned empty output.")

        txt = _strip_think_blocks(txt)
        txt2 = _extract_json_object(txt)

        # Robust: parse first JSON object even if extra data exists
        return _loads_first_json_object(txt2)

    async def complete_json(self, *, instruction: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        payload_txt = _safe_dumps(user_payload)

        base_instr = (instruction or "").strip()
        if base_instr:
            base_instr += "\n\nReturn ONLY valid JSON (no markdown, no extra text)."
        else:
            base_instr = "Return ONLY valid JSON (no markdown, no extra text)."

        last_txt = ""

        # Retry a few times for transient empty outputs (free providers do this sometimes)
        for attempt in range(3):
            # 1) Try with JSON mode
            last_txt = await self._call_chat(
                instruction=base_instr,
                payload_txt=payload_txt,
                max_out=self.max_output_tokens,
                force_json=True,
            )

            # 2) If empty, try again without JSON mode (some providers ignore response_format)
            if not last_txt:
                last_txt = await self._call_chat(
                    instruction=base_instr,
                    payload_txt=payload_txt,
                    max_out=self.max_output_tokens,
                    force_json=False,
                )

            if not last_txt:
                await asyncio.sleep(0.4 * (attempt + 1))
                continue

            cleaned = _strip_think_blocks(last_txt)
            txt2 = _extract_json_object(cleaned)

            try:
                return _loads_first_json_object(txt2)
            except Exception:
                # repair pass (handles truncated / non-json / extra data)
                try:
                    return await self._repair_json(
                        original_instruction=base_instr,
                        bad_txt=last_txt,
                        payload_txt=payload_txt,
                    )
                except Exception as e2:
                    head = (last_txt or "")[:400].replace("\n", "\\n")
                    tail = (last_txt or "")[-200:].replace("\n", "\\n")
                    raise ValueError(f"OpenRouter returned invalid JSON. head={head} ... tail={tail}") from e2

        raise ValueError("OpenRouter returned empty output after retries (no JSON to parse).")