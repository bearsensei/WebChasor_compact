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
from utils.thinking_coordinator import coordinate_unified_response

async def execute_single_query(query: str, router, registry, toolset):
    """执行单个查询的主要逻辑"""
    try:
        # 先路由查询以确定action类型，但只调用一次
        category_enum = await router.classify("", query)
        category = category_enum.value
        print(f"[DEMO][ROUTE] {category}")
        
        # 获取action名称
        action_name = registry.route(category)
        print(f"[DEMO][ACTION] {action_name}")
        
        # 根据配置和action类型决定执行方式
        if action_name == "REASONING" and is_streaming_enabled():
            # 流式输出模式 - 直接调用streaming模块
            print("[DEMO][RESULT] ", end="", flush=True)
            full_content = await stream_response(query)
            if full_content and not get_streaming_format() == "openai":
                print(f"\n[DEMO][FULL_CONTENT] Generated {len(full_content)} characters")
            return full_content
        else:
            # 常规模式 - 直接调用action，避免ChasorCore的重复路由
            action = registry.get(action_name)
            if action:
                # 创建Context对象
                ctx = Context(
                    history="", 
                    query=query, 
                    router_category=category, 
                    hints={}
                )
                # 直接调用action
                result = await action.run(ctx, toolset)
                print(f"[DEMO][RESULT] {result.content}")
                
                # 显示元数据
                if result.meta:
                    print(f"[DEMO][META] fields={len(result.meta)}")
                    if 'search_results_count' in result.meta:
                        print(f"[DEMO][SEARCH] results={result.meta['search_results_count']}")
                    if 'extraction_confidence' in result.meta:
                        print(f"[DEMO][CONFIDENCE] {result.meta['extraction_confidence']:.2f}")
                
                return result.content
            else:
                print(f"[DEMO][ERROR] Action not found: {action_name}")
                return None
        
    except Exception as e:
        print(f"[DEMO][ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None

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
    
    # Demo queries - mix of different types
    queries = [
        "解释一下为什么会有潮汐？",  # Test query
    ]
    
    print("[DEMO][START] WebChasor Demo with Thinking Process")
    print(f"[DEMO][MODEL] synthesizer={synthesizer.model_name}")
    
    # 显示流式输出配置
    streaming_output.print_streaming_info()
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[DEMO][QUERY_{i}] {query}")
        print("-" * 50)
        
        # 使用统一的流式响应
        await coordinate_unified_response(
            query, 
            execute_single_query, 
            query, router, registry, toolset
        )
        
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_demo())