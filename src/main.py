# src/main.py
import asyncio
from artifacts import ActionRegistry
from actions.reasoning import REASONING
from actions.productivity import PRODUCTIVITY
from actions.ir_rag import IR_RAG
from toolset import Toolset
from core import ChasorCore
from stubs import SimpleRouter, SimpleSynthesizer

async def bootstrap():
    # registry
    registry = ActionRegistry()
    for A in (REASONING, PRODUCTIVITY, IR_RAG):
        registry.register(A)

    # tools (only router & synthesizer real for MVP)
    router = SimpleRouter()
    synthesizer = SimpleSynthesizer()
    tools = Toolset(router=router, synthesizer=synthesizer)

    core = ChasorCore(router=router, registry=registry, toolset=tools)

    # tests
    print(await core.run("","Who is LeBron James?"))
    print(await core.run("","Summarize this into bullets: ..."))
    print(await core.run("","Why do teams value spacing in basketball?"))

if __name__ == "__main__":
    asyncio.run(bootstrap())