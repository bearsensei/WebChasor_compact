# main.py
import asyncio
import sys
import os
import json

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from artifacts import ActionRegistry, Context
from synthesizer import Synthesizer
from router import Router  # Use Router instead of SimpleRouter
from actions.productivity import PRODUCTIVITY
from actions.reasoning import REASONING
from actions.ir_rag import IR_RAG, IRConfig
from actions.geo_query import GEO_QUERY
from chasor import ChasorCore
from toolsets import Toolset
from config_manager import get_config
from streaming import stream_response, is_streaming_enabled, get_streaming_format, streaming_output
from utils.timectx import parse_time_intent
from prompt import build_enhancement_instruction

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
    geo_query_action = GEO_QUERY()  # Will read GOOGLE_MAP_KEY from env
    
    # Create registry and register actions
    registry = ActionRegistry()
    registry._reg = {
        "PRODUCTIVITY": productivity_action,
        "REASONING": reasoning_action,
        "IR_RAG": ir_rag_action,
        "INFORMATION_RETRIEVAL": ir_rag_action,  # Map router category to action
        "GEO_QUERY": geo_query_action,
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
        "卢秀燕什么时候可以当台湾领导人？",  # Test query
    ]
    
    print("[DEMO][START] WebChasor Demo")
    print(f"[DEMO][MODEL] synthesizer={synthesizer.model_name}")
    
    # info for streaming output
    streaming_output.print_streaming_info()
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[DEMO][QUERY_{i}] {query}")
        print("-" * 50)
        
        try:
            # route query to determine action type, but only call once
            category_enum = await router.classify("", query)
            category = category_enum.value
            print(f"[DEMO][ROUTE] {category}")
            
            # get action name
            action_name = registry.route(category)
            print(f"[DEMO][ACTION] {action_name}")
            
            # decide execution mode based on config and action type
            if action_name == "REASONING" and is_streaming_enabled():
                # streaming mode - directly call streaming module
                print("[DEMO][RESULT] ", end="", flush=True)
                full_content = await stream_response(query)
                if full_content and not get_streaming_format() == "openai":
                    print(f"\n[DEMO][FULL_CONTENT] Generated {len(full_content)} characters")
            
            else:
                # normal mode - directly call action, avoid ChasorCore's repeated routing
                action = registry.get(action_name)
                if action:
                    # parse time context
                    time_context = parse_time_intent(query)
                    print(f"[DEMO][TIME] intent={time_context.intent}, window={time_context.window}")
                    
                    # create Context object
                    ctx = Context(
                        history="", 
                        query=query, 
                        router_category=category, 
                        hints={},
                        time_context=time_context
                    )
                    # directly call action
                    result = await action.run(ctx, toolset)
                    print(f"[DEMO][RESULT] {result.content}")
                    
                    # show metadata
                    if result.meta:
                        print(f"[DEMO][META] fields={len(result.meta)}")
                        if 'search_results_count' in result.meta:
                            print(f"[DEMO][SEARCH] results={result.meta['search_results_count']}")
                        if 'extraction_confidence' in result.meta:
                            print(f"[DEMO][CONFIDENCE] {result.meta['extraction_confidence']:.2f}")
                else:
                    print(f"[DEMO][ERROR] Action not found: {action_name}")
            
        except Exception as e:
            print(f"[DEMO][ERROR] {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_demo())
