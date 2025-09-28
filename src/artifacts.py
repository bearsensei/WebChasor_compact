# artifacts.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass
class Artifact:
    kind: str                      # "text" | "json" | "table" | "image" | "code" | "math"
    content: Any
    meta: Optional[Dict] = None    # citations, confidence, mime, etc.

@dataclass
class Context:
    history: str
    query: str
    router_category: str           # set by Router
    hints: Dict = None             # optional; e.g., archetype=biography

# action base
class Action:
    name: str = "BASE"
    # Declare capabilities & budgets the core can enforce/log
    requires_tools: List[str] = []         # e.g., ["search","visit","extractor"]
    max_time_s: int = 30
    max_tokens_in: int = 8000
    max_tokens_out: int = 1500

    async def run(self, ctx: Context, toolset: "Toolset") -> Artifact:
        raise NotImplementedError

# registry
class ActionRegistry:
    def __init__(self): self._reg = {}
    def register(self, action_cls: type[Action]): self._reg[action_cls.name]=action_cls()
    def get(self, name: str) -> Action: return self._reg[name]
    def route(self, router_category: str) -> str:
        # mapping is configurable; can live in YAML
        mapping = {
            "INFORMATION_RETRIEVAL": "IR_RAG",
            "TASK_PRODUCTIVITY": "PRODUCTIVITY",
            "KNOWLEDGE_REASONING": "REASONING",
            "CREATIVE_GENERATION": "CREATIVE",
            "MATH_QUERY": "MATH",
            "MULTIMODAL_QUERY": "MULTIMODAL",
            "CONVERSATIONAL_FOLLOWUP": "RESPONSE"
        }
        return mapping.get(router_category, "REASONING")