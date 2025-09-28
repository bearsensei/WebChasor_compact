# evaluator.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EvalReport:
    passed: bool
    score: float
    reasons: Dict[str, float]   # e.g., {"citation_coverage": 0.6, "task_coverage": 1.0}
    suggestions: Dict[str, Any] # machine-usable hints, e.g., {"missing_vars": ["birth"], "need_primary": True}

class Evaluator:
    async def evaluate(self, category: str, plan, artifact, extracted=None) -> EvalReport:
        """
        - category: router category
        - plan: Plan or None
        - artifact: Artifact from Synthesizer
        - extracted: Dict[var -> ExtractedField] (IR only)
        Returns an EvalReport with pass/fail, numeric score, and actionable suggestions.
        """
        raise NotImplementedError