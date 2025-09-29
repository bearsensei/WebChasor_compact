# synthesizer.py
import os
from dataclasses import dataclass
from typing import Optional, Dict
from dotenv import load_dotenv
from config_manager import get_config
from prompt import (
    SYNTHESIZER_GLOBAL_SYSTEM,
    SYNTHESIZER_HIDDEN_REASONING_SCAFFOLD,
    SYNTHESIZER_ACTION_POLICIES,
    SYNTHESIZER_STYLE_PROFILES,
    SYNTHESIZER_PROMPT_TEMPLATE
)

# Load environment variables
load_dotenv()

@dataclass
class StyleProfile:
    name: str
    tone: str            # "friendly", "formal", "analytical"
    persona: str         # "WebChasor—analytical assistant"
    format_prefs: Dict   # {"paragraphs": True, "bullets_max": 7}

def build_prompt(category: str,
                 style: StyleProfile,
                 constraints: Dict,
                 materials: str,
                 task_scaffold: Optional[str] = None) -> str:
    # Language / tone from constraints override profile if present
    lang = constraints.get("language", "auto")
    tone = constraints.get("tone", style.tone)
    reading_level = constraints.get("reading_level", "default")

    # Add internal scaffolding based on category
    internal_scaffold = ""
    if category == "KNOWLEDGE_REASONING":
        internal_scaffold = f"\n# Internal Plan (do not reveal)\n{SYNTHESIZER_HIDDEN_REASONING_SCAFFOLD}\n"
    elif task_scaffold:
        internal_scaffold = f"\n# Internal Task Plan (do not reveal)\n{task_scaffold}\n"

    instruction_hint = constraints.get("instruction_hint", "")
    if instruction_hint:
        instruction_hint = f"\n{instruction_hint}"

    # Use the template from prompt.py
    prompt = SYNTHESIZER_PROMPT_TEMPLATE.format(
        global_system=SYNTHESIZER_GLOBAL_SYSTEM,
        action_policy=SYNTHESIZER_ACTION_POLICIES.get(category, "Provide helpful and accurate responses."),
        persona=style.persona,
        tone=tone,
        reading_level=reading_level,
        language=lang,
        format_prefs=style.format_prefs,
        internal_scaffold=internal_scaffold,
        materials=materials,
        instruction_hint=instruction_hint
    )

    return prompt

class Synthesizer:
    def __init__(self, llm=None):
        """
        Initialize Synthesizer with LLM.
        
        Args:
            llm: Language model function. If None, will try to create from environment variables.
        """
        if llm is None:
            # Try to create LLM from environment variables
            self.llm = self._create_default_llm()
        else:
            self.llm = llm
        
        # Get model configuration from config file
        cfg = get_config()
        self.model_name = cfg.get('models.synthesizer.model_name', 'gpt-4')
        self.temperature = cfg.get('models.synthesizer.temperature', 0.1)
        self.max_tokens = cfg.get('models.synthesizer.max_tokens', 2000)
        
        if cfg.is_decision_logging_enabled('synthesizer'):
            print(f"[SYNTHESIZER][INIT] model={self.model_name}")
    
    def _create_default_llm(self):
        """Create default LLM from environment variables"""
        api_base = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY_AGENT") 
        model = os.getenv("OPENAI_API_MODEL_AGENT_SYNTHESIZER", "gpt-4")
        
        if not api_base or not api_key:
            print("[SYNTHESIZER][WARN] No API credentials, using mock LLM")
            return self._mock_llm
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=api_base)
            
            async def real_llm(prompt, temperature=0):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            return real_llm
            
        except ImportError:
            print("⚠️ OpenAI library not available, using mock LLM")
            return self._mock_llm
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI client: {e}, using mock LLM")
            return self._mock_llm
    
    async def _mock_llm(self, prompt, temperature=0):
        """Fallback mock LLM for testing"""
        if "summarize" in prompt.lower():
            return "• Key point 1: Main topic overview\n• Key point 2: Important details\n• Key point 3: Key conclusions"
        elif "rewrite" in prompt.lower():
            return "The content has been rewritten in a clearer, more accessible style."
        elif "why" in prompt.lower():
            return "This occurs due to multiple interconnected factors including economic, strategic, and operational considerations."
        else:
            return "I've processed your request according to the specified requirements."

    async def generate(self, category, style_key, constraints, materials, task_scaffold=None):
        # Convert style dict to StyleProfile object for compatibility
        style_dict = SYNTHESIZER_STYLE_PROFILES.get(style_key, SYNTHESIZER_STYLE_PROFILES["default_analytical"])
        style = StyleProfile(
            name=style_dict["name"],
            tone=style_dict["tone"],
            persona=style_dict["persona"],
            format_prefs=style_dict["format_prefs"]
        )
        
        prompt = build_prompt(category, style, constraints, materials, task_scaffold)
        # Call your model here (temperature=0 recommended for PRODUCTIVITY)
        text = await self.llm(prompt, temperature=constraints.get("temperature", 0))
        return text
    
    async def synthesize(self, category, plan=None, extracted=None, user_query="", system_prompt=""):
        """
        Synthesize method for compatibility with PRODUCTIVITY and REASONING actions.
        
        Args:
            category: Action category (e.g., "TASK_PRODUCTIVITY")
            plan: Optional plan (not used in current implementation)
            extracted: Optional extracted data (not used in current implementation)
            user_query: User query to process
            system_prompt: System prompt to use
            
        Returns:
            str: Generated response
        """
        # Build a simple prompt combining system and user prompts
        full_prompt = f"{system_prompt}\n\nUser Query: {user_query}"
        
        # Use default temperature for synthesis
        temperature = 0.0 if category == "TASK_PRODUCTIVITY" else 0.1
        
        # Simple decision print
        print(f"[SYNTHESIZER][EXEC] model={self.model_name} temp={temperature} category={category}")
        
        # Call the LLM
        response = await self.llm(full_prompt, temperature=temperature)
        return response