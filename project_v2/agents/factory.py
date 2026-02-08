from __future__ import annotations
from typing import Any, Dict, Optional
import json

from config import MODEL
from agents.instructions import INSTRUCTIONS
from agents.specs import AGENT_SPECS
from agents.llm import LLMProvider, OpenAIJSONLLM

from domain_knowledge.selector import select_domain_knowledge


class AgentFactory:
    def __init__(self, llm: Optional[LLMProvider] = None):
        # default: echtes LLM (JSON-only)
        self.llm = llm or OpenAIJSONLLM(model=MODEL, max_output_tokens=2000)

    def get_instruction_text(self, agent_id: str, meta: Optional[Dict[str, Any]] = None) -> str:
        spec = AGENT_SPECS[agent_id]
        instr = INSTRUCTIONS[spec.instruction_key]
        meta = meta or {}

        fam = str(meta.get("family", "descriptive"))
        allowed = meta.get("allowed_types")

        # Wir machen KEIN instr.format(...) mehr, weil JSON-Beispiele { ... } enthalten.
        # Stattdessen: nur unsere zwei Tokens gezielt ersetzen.
        if isinstance(allowed, list):
            allowed_txt = json.dumps(allowed, ensure_ascii=False)
        else:
            allowed_txt = json.dumps(allowed or [], ensure_ascii=False)

        instr = instr.replace("{family}", fam)
        instr = instr.replace("{allowed_types}", allowed_txt)

        return instr

    async def run(
        self,
        agent_id: str,
        *,
        prompt: str,
        meta: Optional[Dict[str, Any]] = None,
        df_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta_in = meta or {}
        df_profile = df_profile or {}

        # Build a local meta copy so we never mutate caller dictionaries.
        meta = dict(meta_in)

        # Attach compact domain knowledge context automatically.
        # This centralizes DK selection so individual steps do NOT have to call
        # select_domain_knowledge() themselves.
        if "domain_knowledge" not in meta:
            try:
                dk = select_domain_knowledge(
                    step=str(meta.get("step") or agent_id),
                    prompt=prompt,
                    params=meta,  # best-effort: contains family/type/columns when available
                    df_profile=df_profile,
                    top_k=int(meta.get("dk_top_k") or 5),
                    use_cache=True,
                )
            except Exception as e:
                # DK must never break the pipeline. Fall back to empty DK.
                dk = {"snippets": [], "debug": {"selected": 0, "reason": f"dk_error:{type(e).__name__}"}}
            meta["domain_knowledge"] = dk

        spec = AGENT_SPECS[agent_id]
        instr = self.get_instruction_text(agent_id, meta=meta)

        raw = await self.llm.complete_json(
            instruction=instr,
            user_payload={"prompt": prompt, "meta": meta, "df_profile": df_profile},
        )
        return spec.schema_validator(raw)