# pipeline/steps/prompt.py
from __future__ import annotations
from typing import Tuple
import pandas as pd

from pipeline.state import PipelineState

async def step_prompt(*, prompt: str, df: pd.DataFrame, state: PipelineState):
    # Nothing to do; state already has base_prompt
    meta = {"decision": {"prompt": prompt}, "rationale": "Stored initial user prompt."}
    return meta, df