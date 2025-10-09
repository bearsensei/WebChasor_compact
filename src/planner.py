"""
Planner Module
Creates extraction plans from user queries, decomposing them into structured tasks.
"""

import os
import json
import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

from config_manager import get_config
from prompt import PLANNER_PROMPT

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PlanTask:
    """Individual extraction task from planner"""
    fact: str
    variable_name: str
    category: str = "other"
    confidence_threshold: float = 0.7

@dataclass
class ExtractionPlan:
    """Complete plan from planner"""
    archetype: str = "general"
    entity: Optional[str] = None
    tasks_to_extract: List[PlanTask] = field(default_factory=list)
    final_calculation: Optional[Dict[str, Any]] = None
    presentation_hint: Optional[Dict[str, Any]] = None


# ============================================================================
# Planner Class
# ============================================================================

class Planner:
    """Planner component that creates extraction plans"""
    
    def __init__(self, llm_client=None, model_name: str = None):
        """Initialize planner with LLM client"""
        self.client = llm_client
        self.model_name = model_name or get_config().get('models.planner.model_name', 'gpt-4')
        # Import planner prompt
        try:
            self.planner_prompt = PLANNER_PROMPT
        except (ImportError, NameError):
            logger.warning("Could not import PLANNER_PROMPT, using fallback")
            self.planner_prompt = self._get_fallback_prompt()
    
    def _get_fallback_prompt(self) -> str:
        """Fallback planner prompt if import fails"""
        return """
        You are a Task Planner. Decompose the user query into structured extraction tasks.
        Output MUST be valid JSON with this schema:
        {
          "plan": {
            "archetype": "biography|fact_verification|recent_situation|background|comparison|other",
            "entity": "main entity name",
            "tasks_to_extract": [
              {
                "fact": "What specific fact to extract?",
                "variable_name": "snake_case_name",
                "category": "biography|fact_verification|recent_situation|background|comparison|other"
              }
            ]
          }
        }
        """
    
    async def plan(self, query: str, archetype_hint: Optional[str] = None) -> ExtractionPlan:
        """Create extraction plan for the given query"""
        try:
            print(f"📋 PLANNER: Creating plan for query: {query[:100]}...")
            
            if not self.client:
                logger.warning("No LLM client available, using fallback plan")
                return self._create_fallback_plan(query)
            
            # Build planner prompt with query
            prompt = self.planner_prompt.replace("{user_query}", query)
            
            # Call LLM for planning
            cfg = get_config()
            planner_max_tokens = cfg.get('models.planner.max_tokens', 2000)
            planner_temperature = cfg.get('models.planner.temperature', 0.1)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"User: {query}"}
                ],
                temperature=planner_temperature,
                max_tokens=planner_max_tokens
            )
            
            plan_text = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason

            # Check if response was truncated
            if finish_reason == 'length':
                logger.warning(f"Planner response was truncated (finish_reason=length), may cause JSON parse error")
                print(f"[PLANNER][WARN] Response truncated at {len(plan_text)} chars, consider increasing max_tokens")
            
            # Parse JSON response
            try:
                plan_data = json.loads(plan_text)
                plan_dict = plan_data.get("plan", plan_data)
                
                # Convert to structured plan
                tasks = []
                for task_data in plan_dict.get("tasks_to_extract", []):
                    tasks.append(PlanTask(
                        fact=task_data["fact"],
                        variable_name=task_data["variable_name"],
                        category=task_data.get("category", "other")
                    ))
                
                extraction_plan = ExtractionPlan(
                    archetype=plan_dict.get("archetype", "general"),
                    entity=plan_dict.get("entity"),
                    tasks_to_extract=tasks,
                    final_calculation=plan_dict.get("final_calculation"),
                    presentation_hint=plan_dict.get("presentation_hint")
                )
                
                print(f"📋 PLANNER: Created plan with {len(tasks)} tasks, archetype: {extraction_plan.archetype}")
                return extraction_plan
                
            except json.JSONDecodeError as e:
                print(f"[PLANNER][ERROR] Failed to parse planner JSON: {e}")
                print(f"[PLANNER][ERROR] Raw response (first 500 chars): {plan_text[:500]}")
                print(f"[PLANNER][ERROR] Raw response (last 200 chars): {plan_text[-200:]}")
                logger.error(f"Failed to parse planner JSON: {e}")
                logger.error(f"Raw response: {plan_text}")
                print(f"[PLANNER][FALLBACK] Using fallback plan")
                return self._create_fallback_plan(query)
                
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return self._create_fallback_plan(query)
    
    def _create_fallback_plan(self, query: str) -> ExtractionPlan:
        """Create a simple fallback plan when LLM planning fails"""
        # Extract key terms for basic planning - support both English and Chinese
        english_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        chinese_terms = re.findall(r'[\u4e00-\u9fff]+', query)
        
        # Prefer Chinese terms for Chinese queries, English terms for English queries
        if chinese_terms:
            entity = chinese_terms[0]  # Use first Chinese term
        elif english_terms:
            entity = english_terms[0]  # Use first English term
        else:
            entity = "unknown"
        
        # Create basic extraction tasks with appropriate language
        if chinese_terms:
            # Use Chinese for Chinese queries
            tasks = [
                PlanTask(
                    fact=f"关于{entity}的基本信息是什么？",
                    variable_name="basic_info",
                    category="background"
                ),
                PlanTask(
                    fact=f"关于{entity}的最新信息有哪些？",
                    variable_name="key_facts",
                    category="recent_situation"
                )
            ]
        else:
            # Use English for English queries
            tasks = [
                PlanTask(
                    fact=f"What is the basic information about {entity}?",
                    variable_name="basic_info",
                    category="background"
                ),
                PlanTask(
                    fact=f"What are the recent facts about {entity}?",
                    variable_name="key_facts",
                    category="recent_situation"
                )
            ]
        
        return ExtractionPlan(
            archetype="general",
            entity=entity,
            tasks_to_extract=tasks
        )


# ============================================================================
# Test/Demo Code (runs when executed directly)
# ============================================================================

if __name__ == "__main__":
    import time
    import openai
    from utils.timectx import parse_time_intent
    
    load_dotenv()
    
    # Load configuration
    cfg = get_config()
    OPENAI_API_BASE = cfg.get('external_services.openai.api_base', 'https://api.openai.com/v1')
    OPENAI_API_KEY_AGENT = os.getenv("OPENAI_API_KEY_AGENT")
    OPENAI_API_MODEL_AGENT_PLANNER = cfg.get('models.planner.model_name', 'gpt-4')
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY_AGENT, base_url=OPENAI_API_BASE)
    
    # Test query
    USER_INPUT = "郭毅可什么时候可以当浸会大学校长？"
    
    print(f"[TEST][Planner] Testing with query: {USER_INPUT}")
    print("=" * 80)
    
    # Initialize planner
    planner = Planner(llm_client=client, model_name=OPENAI_API_MODEL_AGENT_PLANNER)
    
    # Create plan
    time_start = time.time()
    
    # Note: The async method needs to be run in an event loop
    import asyncio
    plan = asyncio.run(planner.plan(USER_INPUT))
    
    time_end = time.time()
    
    # Display results
    print("\n" + "=" * 80)
    print("[TEST][Result] Plan created successfully!")
    print(f"[TEST][Performance] Time taken: {time_end - time_start:.2f} seconds")
    print("\n" + "=" * 80)
    print(f"[TEST][Plan] Archetype: {plan.archetype}")
    print(f"[TEST][Plan] Entity: {plan.entity}")
    print(f"[TEST][Plan] Number of tasks: {len(plan.tasks_to_extract)}")
    print("\n[TEST][Tasks]:")
    for i, task in enumerate(plan.tasks_to_extract, 1):
        print(f"  {i}. {task.variable_name} ({task.category})")
        print(f"     Fact: {task.fact}")
    print("=" * 80)
