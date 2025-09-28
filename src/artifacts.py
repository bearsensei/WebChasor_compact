# artifacts.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from toolsets import Toolset

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
            "INFORMATION_RETRIEVAL": "IR_RAG",     # Future: Information Retrieval with RAG
            "TASK_PRODUCTIVITY": "PRODUCTIVITY",
            "KNOWLEDGE_REASONING": "REASONING",
            "CREATIVE_GENERATION": "CREATIVE",     # Future: Creative content generation
            "MATH_QUERY": "MATH",                  # Future: Mathematical computation
            "MULTIMODAL_QUERY": "MULTIMODAL",      # Future: Image/video processing
            "CONVERSATIONAL_FOLLOWUP": "RESPONSE"  # Future: Conversational responses
        }
        
        action_name = mapping.get(router_category, "REASONING")
        
        # Check if the action is available in registry, fallback to available ones
        if action_name not in self._reg:
            # Future actions not yet implemented, use available fallbacks
            fallback_mapping = {
                "IR_RAG": "REASONING",           # Information retrieval → reasoning
                "CREATIVE": "REASONING",         # Creative tasks → reasoning  
                "MATH": "REASONING",             # Math queries → reasoning
                "MULTIMODAL": "REASONING",       # Multimodal → reasoning
                "RESPONSE": "REASONING"          # Responses → reasoning
            }
            original_action = action_name
            action_name = fallback_mapping.get(action_name, "REASONING")
            
            # Simple decision print
            print(f"📋 REGISTRY DECISION: {router_category} → {original_action} → {action_name} (fallback)")
        else:
            # Simple decision print
            print(f"📋 REGISTRY DECISION: {router_category} → {action_name} (direct)")
        
        return action_name