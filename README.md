# Hybrid Multi-Agent LLM Data Agent (Deterministic Analytics Pipeline)

This repo provides a **hybrid, decision-driven analytics pipeline** for automated business insights from tabular data (CSV / Pandas DataFrame).

**Core idea:** the LLM is limited to **bounded, structured decisions** (routing, scoping, column roles, etc.), while **all computations and plots run deterministically** in Python. This improves **reproducibility, reviewability, and auditability** via explicit artifacts and step-wise traces.

---

## What’s inside

- **Deterministic pipeline engine** with step-wise trace (`project_v2/pipeline/`)
- **Analysis templates**: descriptive / diagnostic / predictive / prescriptive (`project_v2/analysis/`)
- **Pluggable LLM backends**: OpenAI / OpenRouter / Ollama (`project_v2/agents/`)
- **Domain knowledge ingestion**: PDF/text loaders + selector (`project_v2/domain_knowledge/`)
- **Experiment notebooks & results**: performance & stability studies (`experiments/`)

---

## Repository structure

- `project_v2/` — main code
  - `data_agent_template.ipynb` — minimal usage template (recommended starting point)
  - `agents/` — LLM abstraction + backends (`llm.py`, `llm_openrouter.py`, `llm_ollama.py`, `factory.py`)
  - `pipeline/` — orchestration, step registry, reporting, review flow (`steps/` contains each step)
  - `analysis/` — deterministic analysis templates by family
  - `domain_knowledge/` — loaders + store for dataset card / notes (PDF/text)
  - `transforms/`, `viz/` — deterministic transforms + plotting utilities
  - `run_reports/`, `artifacts/` — generated outputs (reports, images)
- `experiments/` — study artifacts and aggregated results
  - `performance study/` — PPTX evaluator packs (human rating)
  - `stability study/` — stability result spreadsheets
  - `performance_study_results.xlsx` — aggregated performance results

> Note: folders with spaces are fine, but can be slightly annoying in shells. Rename if you prefer.

---

## Requirements

- Python **3.10+** (recommended)
- Optional: Jupyter (for notebooks)
- One API key (depending on backend):
  - OpenAI: `OPENAI_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY`

Install dependencies:

```bash
pip install -r requirements.txt
```

### What the `requirements.txt` covers

Your environment scan included **stdlib** modules (e.g. `asyncio`, `datetime`, `json`, `re`) and **local modules** (e.g. `analysis`, `config`, `domain_knowledge`, `viz`).  
Those do **not** belong in `requirements.txt`.

The file focuses on **third‑party packages** detected in the scan (pinned for reproducibility) plus **runtime packages used by optional backends/loaders** (`openai`, `PyPDF2`), even though the scan didn’t find their distribution metadata.

---

## Quickstart (template notebook)

Open `project_v2/data_agent_template.ipynb` and run something like:

```python
# Jupyter
%matplotlib inline

import os
import pandas as pd

from project_v2.pipeline.engine import run_pipeline
from project_v2.pipeline.reporting import show_run_report
from project_v2.pipeline.review import ReviewConfig

# Load data (no parse_dates)
file_path = r"path_to_file.csv"
df = pd.read_csv(file_path, sep=",", on_bad_lines="skip", engine="python").reset_index(drop=True)

# Provide an API key via environment (recommended to set outside the notebook)
# Example:
# os.environ["OPENAI_API_KEY"] = "..."
# or:
# os.environ["OPENROUTER_API_KEY"] = "..."

prompt = "Show me the distribution of ta rel, only finished parts"

state, df_out, results = await run_pipeline(
    prompt=prompt,
    df=df,
    review_config=ReviewConfig(
        enabled=True,
        # Example: pause after certain steps for review
        # after_steps={"family","type","filters","columns","prepare","aggregate","viz","analyze","verify","finalize"},
        after_steps=set(),
        show_step_inputs=True,
    ),
)

show_run_report(
    results,
    show_head_df=df_out,
    show_step_inputs=False,  # avoid duplicating step inputs
    verbose_steps=False,     # no large step dumps at the end
    render_final=True,       # plots + final text
    final_head_rows=5,
)
```

---

## How it works (short)

1. **LLM decision layer (JSON artifacts)** produces bounded decisions, e.g.:
   - analysis family/type, filters, column roles, preparation/aggregation intent, viz metadata
2. **Deterministic execution layer** applies:
   - filtering, preparation, aggregation, template analysis, plotting, verification
3. **Outputs** include:
   - evidence (tables/plots), grounded narrative, and a step-wise trace/run report

---

## Experiments (optional)

Reproduction notebooks live in `project_v2/` (e.g., `eval_exp_perform.ipynb`, `eval_exp_stability.ipynb`).

Aggregated results and evaluator packs are stored in `experiments/`.

---
