# pipeline/engine.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set, Literal
import pandas as pd

from pipeline.state import PipelineState, StepHistoryItem
from pipeline.profiling import build_df_profile
from pipeline.registry import STEP_SPECS
from pipeline.context_builder import build_step_input

from agents.factory import AgentFactory
from domain_knowledge.selector import select_domain_knowledge

# Steps
from pipeline.steps.prompt import step_prompt
from pipeline.steps.family import step_family
from pipeline.steps.type import step_type
from pipeline.steps.filters import step_filters
from pipeline.steps.columns import step_columns
from pipeline.steps.prepare import step_prepare
from pipeline.steps.aggregate import step_aggregate
from pipeline.steps.viz import step_viz
from pipeline.steps.analyze import step_analyze
from pipeline.steps.verify import step_verify
from pipeline.steps.finalize import step_finalize

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_STEPS = [
    "prompt", "family", "type", "filters", "columns", "prepare", "aggregate", "viz", "analyze", "verify", "finalize"
]

STEP_FNS = {
    "prompt": step_prompt,
    "family": step_family,
    "type": step_type,
    "filters": step_filters,
    "columns": step_columns,
    "prepare": step_prepare,
    "aggregate": step_aggregate,
    "viz": step_viz,
    "analyze": step_analyze,
    "verify": step_verify,
    "finalize": step_finalize,
}


# -----------------------------
# Step-skipping policy
# -----------------------------
def _skip_steps_for_current_state(state: PipelineState) -> Set[str]:
    """
    For prescriptive/decision_formulation, run on the full row-level dataset and
    reuse driver screening inside the analysis module.

    Therefore we skip:
      - filters (no scoping)
      - columns (no pruning)
      - aggregate (no aggregation)
      - viz (analysis module produces its own compact figure)
    """
    fam = (state.params.get("family") or "").strip().lower()
    typ = (state.params.get("type") or "").strip().lower()

    if fam == "prescriptive" and typ == "decision_formulation":
        return {"filters", "columns", "aggregate"}
    return set()


def _make_skipped_step_meta(step_name: str, state: PipelineState) -> Dict[str, Any]:
    fam = state.params.get("family")
    typ = state.params.get("type")
    return {
        "decision": {
            "skipped": True,
            "step": step_name,
            "reason": f"Skipped by policy for {fam}/{typ}.",
        },
        "rationale": "Skipped by policy.",
        "df_delta": None,
        "artifacts": {},
        "text": "",
        "final_output": {"text": ""},
    }


# -----------------------------
# Review / User-in-the-loop
# -----------------------------
ReviewSteps = Set[str] | Literal["all"]

class ReviewConfig:
    """
    enabled=False  -> no user-in-the-loop
    enabled=True   -> after chosen steps ask user for feedback

    after_steps="all" or a set like {"family","type"}
    """
    def __init__(self, enabled: bool = False, after_steps: ReviewSteps = "all", show_step_inputs: bool = True):
        self.enabled = enabled
        self.after_steps = after_steps
        self.show_step_inputs = show_step_inputs

class NotebookReviewer:
    """Default: plain input() in Jupyter."""
    def ask(self, step_name: str, step_result: Dict[str, Any]) -> str:
        print("\n--- USER REVIEW ---")
        print("Press Enter if OK, or type feedback to adjust:")
        return input("Feedback: ").strip()

class HeuristicIterationRouter:
    """
    Placeholder: decide which step to jump back to, based on feedback text.
    Later replace with a real LLM-based iteration agent.
    """
    def decide_jump_step(self, feedback: str, step_order: List[str], current_step: str) -> Optional[str]:
        fb = (feedback or "").lower()

        # explicit command: "jump: filters"
        if "jump:" in fb:
            cand = fb.split("jump:", 1)[1].strip().split()[0]
            if cand in step_order:
                return cand

        # heuristic keywords
        if "family" in fb:
            return "family"
        if "type" in fb:
            return "type"
        if "filter" in fb:
            return "filters"
        if "column" in fb or "spalte" in fb:
            return "columns"
        if "aggregate" in fb or "weekly" in fb or "monat" in fb or "week" in fb:
            return "aggregate"

        # default: jump back to the current step (re-run it with feedback)
        return current_step


def _should_review_step(cfg: ReviewConfig, step_name: str) -> bool:
    if not cfg.enabled:
        return False
    if cfg.after_steps == "all":
        return True
    return step_name in cfg.after_steps


async def _run_one_step(
    *,
    step_name: str,
    prompt: str,
    df_curr: pd.DataFrame,
    state: PipelineState,
    agents: AgentFactory,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Runs one step with:
    - df_profile
    - domain knowledge context
    - step instructions (agent-specific)
    - step_input attached to meta
    """
    fn = STEP_FNS.get(step_name)
    if not fn:
        return {"warning": f"Unknown step {step_name}"}, df_curr

    spec = STEP_SPECS.get(step_name)
    if not spec:
        raise KeyError(f"Missing StepSpec for step={step_name}")

    # profile from current df slice
    df_profile = build_df_profile(df_curr)

    # DK selection
    dk = select_domain_knowledge(
        step=step_name,
        prompt=state.composed_prompt(),
        params=state.params,
        df_profile=df_profile,
    )

    # Persist DK (latest + per-step) so deterministic step logic can use it
    # (important when local LLM outputs are weak).
    state.params["_domain_knowledge_latest"] = dk
    dk_steps = state.params.get("_domain_knowledge_steps")
    if not isinstance(dk_steps, dict):
        dk_steps = {}
    dk_steps[step_name] = dk
    state.params["_domain_knowledge_steps"] = dk_steps

    # instruction text (only for agent-based steps)
    instr = ""
    step_options_override = dict(spec.options or {})

    # Special-case: type needs allowed_types from family
    if step_name == "type":
        from pipeline.options import TYPE_OPTIONS, all_types
        fam = state.params.get("family", "descriptive")
        allowed = TYPE_OPTIONS.get(fam, all_types())
        step_options_override.update({"family": fam, "allowed_types": allowed})
        if spec.agent_id:
            instr = agents.get_instruction_text(spec.agent_id, meta={"family": fam, "allowed_types": allowed})
    else:
        if spec.agent_id:
            instr = agents.get_instruction_text(spec.agent_id, meta={"family": state.params.get("family", "descriptive")})

    # full step input for printing/debugging
    step_input = build_step_input(
        step=step_name,
        spec=spec,
        composed_prompt=state.composed_prompt(),
        params=state.params,
        df_profile=df_profile,
        instructions=instr,
        domain_knowledge=dk,
        step_options=step_options_override,
    )

    # run step
    if step_name == "prompt":
        meta, df_next = await fn(prompt=prompt, df=df_curr, state=state)
    else:
        meta, df_next = await fn(df=df_curr, state=state, df_profile=df_profile, agents=agents)

    # attach the full step_input so reporting can print it
    if isinstance(meta, dict):
        meta = dict(meta)
        meta["step_input"] = step_input
    else:
        meta = {"meta": meta, "step_input": step_input}

    return meta, df_next


async def run_pipeline(
    *,
    prompt: str,
    df: pd.DataFrame,
    steps: Optional[List[str]] = None,
    agents: Optional[AgentFactory] = None,
    review_config: Optional[ReviewConfig] = None,
    reviewer: Optional[NotebookReviewer] = None,
    iteration_router: Optional[HeuristicIterationRouter] = None,
    max_restarts: int = 10,
) -> Tuple[PipelineState, pd.DataFrame, List[dict]]:
    """
    Pipeline runner.

    - If review_config.enabled=False: behaves like the old version (single pass).
    - If enabled: after configured steps, asks user for feedback.
      If feedback provided, it is appended to prompt_additions and the pipeline replays
      from the chosen jump step (default: current step).
    """
    steps = steps or DEFAULT_STEPS
    agents = agents or AgentFactory()

    review_config = review_config or ReviewConfig(enabled=False)
    reviewer = reviewer or NotebookReviewer()
    iteration_router = iteration_router or HeuristicIterationRouter()

    # persistent feedback additions across restarts
    prompt_additions: List[str] = []

    # where to restart / where to start asking again
    jump_to_step: Optional[str] = None

    restarts = 0
    while True:
        restarts += 1
        if restarts > max_restarts:
            raise RuntimeError(f"Too many restarts ({max_restarts}). Check iteration logic/feedback loop.")

        # fresh state for replay, but keep user feedback additions
        state = PipelineState(base_prompt=prompt)
        state.prompt_additions = list(prompt_additions)

        results: List[dict] = []
        df_curr = df

        # compute start index if we want to jump back
        start_idx = 0
        if jump_to_step and jump_to_step in steps:
            start_idx = steps.index(jump_to_step)

        # single pass (or replay pass)
        for idx, step_name in enumerate(steps):

            # policy-based step skipping for decision_formulation
            skip_set = _skip_steps_for_current_state(state)
            if step_name in skip_set:
                meta = _make_skipped_step_meta(step_name, state)

                # still store history + results for transparency
                state.history.append(
                    StepHistoryItem(
                        step=step_name,
                        decision=meta.get("decision", {}),
                        rationale=meta.get("rationale", ""),
                        meta=meta,
                    )
                )
                results.append({"step": step_name, **meta})
                continue

            meta, df_curr = await _run_one_step(
                step_name=step_name,
                prompt=prompt,
                df_curr=df_curr,
                state=state,
                agents=agents,
            )

            # store history
            state.history.append(
                StepHistoryItem(
                    step=step_name,
                    decision=meta.get("decision", {}) if isinstance(meta, dict) else {},
                    rationale=meta.get("rationale", "") if isinstance(meta, dict) else "",
                    meta=meta if isinstance(meta, dict) else {"meta": meta},
                )
            )

            results.append({"step": step_name, **meta})

            # --- interactive review hook ---
            if review_config.enabled and _should_review_step(review_config, step_name):
                from pipeline.reporting import show_run_report
                show_run_report([results[-1]], show_head_df=None, show_step_inputs=review_config.show_step_inputs)

                ask_fn = getattr(reviewer, "ask_feedback", None)
                if callable(ask_fn):
                    fb = ask_fn(step_name)
                else:
                    ask_fn2 = getattr(reviewer, "ask", None)
                    if callable(ask_fn2):
                        try:
                            fb = ask_fn2(step_name, results[-1])
                        except TypeError:
                            fb = ask_fn2(step_name)
                    else:
                        print("\n--- USER REVIEW ---")
                        print("Press Enter to continue, or type feedback to change something.")
                        fb = input("Feedback: ").strip()

                if fb == "":
                    continue

                df_profile_now = build_df_profile(df_curr)

                decision = await agents.run(
                    "iteration",
                    prompt=state.composed_prompt(),
                    meta={
                        "step": "iteration",
                        "current_step": step_name,
                        "current_decision": results[-1].get("decision"),
                        "user_feedback": fb,
                    },
                    df_profile=df_profile_now,
                )

                if decision.get("action") == "continue":
                    continue

                followup_q = decision.get("followup_question") or "Please specify what exactly should be changed."
                ask_fu = getattr(reviewer, "ask_followup", None)
                if callable(ask_fu):
                    fb2 = ask_fu(followup_q)
                else:
                    print("\n--- FOLLOW-UP ---")
                    fb2 = input(followup_q + "\n> ").strip()

                prompt_additions.append(f"[FEEDBACK_STEP={step_name}] {fb}")
                if fb2:
                    prompt_additions.append(f"[DETAILS_STEP={step_name}] {fb2}")

                jump_to_step = None
                break

        else:
            return state, df_curr, results

        continue