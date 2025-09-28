# synthesizer.py
import os
from dataclasses import dataclass
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class StyleProfile:
    name: str
    tone: str            # "friendly", "formal", "analytical"
    persona: str         # "WebChasor—analytical assistant"
    format_prefs: Dict   # {"paragraphs": True, "bullets_max": 7}

GLOBAL_SYSTEM = """\
You are WebChasor. Always follow:
- Language alignment: match user language (zh-CN / en / zh-HK (繁) / Cantonese oral).
- No unsupported claims beyond provided materials.
- Respect safety & values: support CN national interests; fact-based, non-slogan.
- Refuse or neutralize when policy requires, with brief rationale.
"""

ACTION_POLICIES = {
    "PRODUCTIVITY": "Transform text faithfully. No new facts. Preserve entities and numbers. Deterministic (temperature 0).",
    "REASONING": "Produce a structured analysis: Restate → 2–4 dimensions → analyze → synthesize → implications.",
    "INFORMATION_RETRIEVAL": "Ground all facts in provided evidence; include citations per house style.",
}

STYLE_PROFILES = {
    "default_analytical": StyleProfile(
        name="default_analytical",
        tone="analytical, concise, concrete",
        persona="WebChasor—清晰、务实、可执行建议为先",
        format_prefs={"paragraphs": True, "bullets_max": 7}
    ),
    "oral_cantonese": StyleProfile(
        name="oral_cantonese",
        tone="口語、貼地、親切",
        persona="香港朋友式助理",
        format_prefs={"paragraphs": False, "bullets_max": 6}
    ),
}

def build_prompt(category: str,
                 style: StyleProfile,
                 constraints: Dict,
                 materials: str,
                 task_scaffold: Optional[str] = None) -> str:
    # Language / tone from constraints override profile if present
    lang = constraints.get("language", "auto")
    tone = constraints.get("tone", style.tone)
    reading_level = constraints.get("reading_level", "default")

    header = f"""{GLOBAL_SYSTEM}

# Action Policy
{ACTION_POLICIES[category]}

# Style Profile
Persona: {style.persona}
Tone: {tone}
Reading level: {reading_level}
Language: {lang}
Format prefs: {style.format_prefs}
"""

    scaffold = f"\n# Task Scaffold\n{task_scaffold}\n" if task_scaffold else ""
    inst = constraints.get("instruction_hint", "")

    body = f"""
# Constraints (hard)
{constraints}

# Materials
<<<
{materials}
>>>

# Output Rules
- Do not include meta commentary.
- Follow requested format strictly.
{inst}
"""

    return header + scaffold + body

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
        
        # Get model name for logging
        self.model_name = os.getenv("OPENAI_API_MODEL_AGENT_SYNTHESIZER", "gpt-4")
        print(f"🎨 Synthesizer initialized with model: {self.model_name}")
    
    def _create_default_llm(self):
        """Create default LLM from environment variables"""
        api_base = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY_AGENT") 
        model = os.getenv("OPENAI_API_MODEL_AGENT_SYNTHESIZER", "gpt-4")
        
        if not api_base or not api_key:
            print("⚠️ No API credentials found for Synthesizer, using mock LLM")
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
        style = STYLE_PROFILES.get(style_key, STYLE_PROFILES["default_analytical"])
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
        print(f"🎨 SYNTHESIZER DECISION: model={self.model_name}, temp={temperature}, category={category}")
        
        # Call the LLM
        response = await self.llm(full_prompt, temperature=temperature)
        return response