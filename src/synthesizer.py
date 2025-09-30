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
    render_synthesizer_prompt
)
from utils.lang_detect import detect_lang_4way
from utils.timectx import TimeContext
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
                 task_scaffold: Optional[str] = None,
                 time_context: Optional[TimeContext] = None) -> str:
    
        # 添加时间上下文到提示词
    time_hint = ""
    if time_context:
        if time_context.intent == "latest":
            time_hint = f"\n\n时间上下文：用户询问最新信息（截至 {time_context.display_cutoff}）"
        elif time_context.intent == "trend":
            time_hint = f"\n\n时间上下文：用户询问趋势分析（{time_context.window[0]} 至 {time_context.window[1]}）"
        elif time_context.intent == "historic":
            time_hint = f"\n\n时间上下文：用户询问历史信息（{time_context.window[0]} 至 {time_context.window[1]}）"
    
    # 在 materials 中添加时间信息
    materials = materials + time_hint

    # 直接使用 render_synthesizer_prompt 函数
    return render_synthesizer_prompt(
        action_policy=SYNTHESIZER_ACTION_POLICIES.get(category, "Provide helpful and accurate responses."),
        materials=materials,
        user_query=materials,  # 将 materials 作为 user_query 传递用于语言检测
        language=constraints.get("language"),
        reading_level=constraints.get("reading_level", "general"),
        preferred_style=None,  # 让函数自动检测样式
        global_system=SYNTHESIZER_GLOBAL_SYSTEM,
        internal_scaffold=SYNTHESIZER_HIDDEN_REASONING_SCAFFOLD if category == "KNOWLEDGE_REASONING" else (task_scaffold or "")
    )

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
        api_base = get_config().get('external_services.openai.api_base', 'https://api.openai.com/v1')
        api_key = os.getenv("OPENAI_API_KEY_AGENT") 
        model = get_config().get('models.synthesizer.model_name', 'gpt-4')
        
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
        # 直接使用 render_synthesizer_prompt，它会自动处理样式选择和语言检测
        

        auto_lang = detect_lang_4way(materials) # 若没有 user_query，可退回 materials
        print(f'[SYNTHESIZER][LANG_DEBUG] auto_lang: {auto_lang}')
        prompt = render_synthesizer_prompt(
            action_policy=SYNTHESIZER_ACTION_POLICIES.get(category, "Provide helpful and accurate responses."),
            materials=materials,
            user_query=materials,
            language=auto_lang,
            reading_level=constraints.get("reading_level", "general"),
            preferred_style=(style_key if style_key and style_key != "auto" else None),
            global_system=SYNTHESIZER_GLOBAL_SYSTEM,
            internal_scaffold=(SYNTHESIZER_HIDDEN_REASONING_SCAFFOLD if category == "KNOWLEDGE_REASONING" else "")
        )

        temperature = 0.0 if category == "TASK_PRODUCTIVITY" else constraints.get("temperature", 0.1)
        print(f"[SYNTHESIZER][EXEC] model={self.model_name} temp={temperature} category={category} lang={auto_lang}")

        # 3) 把渲染后的 prompt 交给模型
        response = await self.llm(prompt, temperature=temperature)
        return response
            
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