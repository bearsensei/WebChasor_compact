# main.py
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from artifacts import ActionRegistry, Context
from synthesizer import Synthesizer
from router import Router  # Use Router instead of SimpleRouter
from actions.productivity import PRODUCTIVITY
from actions.reasoning import REASONING
from actions.ir_rag import IR_RAG, IRConfig
from chasor import ChasorCore
from toolsets import Toolset
from config_manager import get_config

async def run_demo():
    # Load and display configuration
    cfg = get_config()
    print(f"[CONFIG][LOADED] {cfg}")
    print(f"[CONFIG][WEB_SCRAPING] {cfg.get('ir_rag.web_scraping.enabled', False)}")
    
    # Initialize components - They will auto-configure from YAML
    synthesizer = Synthesizer()
    router = Router()
    
    # Create toolset with synthesizer
    toolset = Toolset(
        router=router,
        synthesizer=synthesizer
    )
    
    # Initialize actions - They will auto-configure from YAML
    productivity_action = PRODUCTIVITY()
    reasoning_action = REASONING()
    
    # Create OpenAI client for IR_RAG if credentials are available
    llm_client = None
    api_base = get_config().get('external_services.openai.api_base')
    api_key = os.getenv("OPENAI_API_KEY_AGENT")
    
    if api_base and api_key:
        try:
            import openai
            llm_client = openai.OpenAI(api_key=api_key, base_url=api_base)
            print(f"[IR_RAG][INIT] OpenAI client ready")
        except ImportError:
            print("[IR_RAG][WARN] OpenAI library not available, using fallback")
    else:
        print("[IR_RAG][WARN] No API credentials, using fallback")
    
    # IR_RAG will auto-configure from YAML
    ir_rag_action = IR_RAG(llm_client=llm_client)
    
    # Create registry and register actions
    registry = ActionRegistry()
    registry._reg = {
        "PRODUCTIVITY": productivity_action,
        "REASONING": reasoning_action,
        "IR_RAG": ir_rag_action,
        "INFORMATION_RETRIEVAL": ir_rag_action,  # Map router category to action
    }
    
    # Initialize ChasorCore with correct parameters
    core = ChasorCore(
        router=router, 
        registry=registry, 
        toolset=toolset,
        evaluator=None  # Add evaluator parameter if needed
    )
    
    # Demo queries - mix of different types
    queries = [
        "介绍叶玉如校长",  # IR RAG
    ]
    
    print("[DEMO][START] WebChasor Demo")
    print(f"[DEMO][MODEL] synthesizer={synthesizer.model_name}")
    print("[DEMO][ACTIONS] available=PRODUCTIVITY,REASONING,IR_RAG")
    print("[DEMO][ACTIONS] future=CREATIVE,MATH,MULTIMODAL,RESPONSE")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[DEMO][QUERY_{i}] {query}")
        print("-" * 50)
        
        try:
            # Run the query through ChasorCore
            result = await core.run("", query)
            
            print(f"[DEMO][RESULT] {result.content}")
            if result.meta:
                print(f"[DEMO][META] fields={len(result.meta)}")
                if 'search_results_count' in result.meta:
                    print(f"[DEMO][SEARCH] results={result.meta['search_results_count']}")
                if 'extraction_confidence' in result.meta:
                    print(f"[DEMO][CONFIDENCE] {result.meta['extraction_confidence']:.2f}")
            
        except Exception as e:
            print(f"[DEMO][ERROR] {e}")
        
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_demo())