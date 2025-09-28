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
from chasor import ChasorCore
from toolsets import Toolset

async def run_demo():
    # Initialize components - Synthesizer will auto-configure from environment variables
    synthesizer = Synthesizer()  # Will use OPENAI_API_MODEL_AGENT_SYNTHESIZER
    router = Router()  # Use the correct Router class
    
    # Create toolset with synthesizer
    toolset = Toolset(
        router=router,
        synthesizer=synthesizer
    )
    
    # Initialize actions with proper configuration
    productivity_action = PRODUCTIVITY()
    reasoning_action = REASONING()
    
    # Create registry and register actions
    registry = ActionRegistry()
    registry._reg = {
        "PRODUCTIVITY": productivity_action,
        "REASONING": reasoning_action,
    }
    
    # Initialize ChasorCore with correct parameters
    core = ChasorCore(
        router=router, 
        registry=registry, 
        toolset=toolset,
        evaluator=None  # Add evaluator parameter if needed
    )
    
    # Demo queries
    queries = [
        "Why do some companies thrive during recessions while others collapse, even in the same industry?"
    ]
    
    print("🚀 Starting WebChasor Demo...")
    print("📋 Available Actions: PRODUCTIVITY, REASONING")
    print("🔮 Future Actions: IR_RAG, CREATIVE, MATH, MULTIMODAL, RESPONSE")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        try:
            print(f"\n📝 Query {i}: {query}")
            print("-" * 40)
            
            artifact = await core.run(history="", user_query=query)
            
            print(f"📊 Result Type: {artifact.kind}")
            print(f"📄 Content:")
            print(artifact.content)
            
            if artifact.meta:
                print(f"🔍 Metadata:")
                for key, value in artifact.meta.items():
                    if key in ['tokens_in', 'tokens_out', 'task', 'language', 'tone']:
                        print(f"  {key}: {value}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error processing query {i}: {str(e)}")
            print(f"   Query: {query}")
            print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_demo())