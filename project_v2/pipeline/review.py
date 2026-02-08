# pipeline/review.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Set, Literal, Dict, Any

ReviewSteps = Set[str] | Literal["all"]

@dataclass
class ReviewConfig:
    enabled: bool = False
    after_steps: ReviewSteps = "all"
    show_step_inputs: bool = True

class NotebookReviewer:
    """
    Backwards compatible reviewer:
    - engine may call ask_feedback()
    - older code may call ask()
    """
    def ask_feedback(self, step_name: str) -> str:
        print("\n--- USER REVIEW ---")
        print("Press Enter to continue, or type feedback to change something.")
        return input("Feedback: ").strip()

    def ask_followup(self, question: str) -> str:
        q = (question or "").strip() or "Please specify what exactly should be changed."
        print("\n--- FOLLOW-UP ---")
        return input(q + "\n> ").strip()

    # compatibility alias
    def ask(self, step_name: str, step_result: Dict[str, Any]) -> str:
        return self.ask_feedback(step_name)