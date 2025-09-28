# src/actions/ir_rag.py
from artifacts import Action, Artifact, Context

class IR_RAG(Action):
    name = "IR_RAG"
    requires_tools = ["planner","serp","visitor","extractor","synthesizer"]

    async def run(self, ctx: Context, toolset) -> Artifact:
        # MVP: just echo to synthesizer; later plug your real IR pipeline here
        text = await toolset.synthesizer.synthesize("INFORMATION_RETRIEVAL", None, None,
                                                    f"(IR stub) {ctx.query}")
        return Artifact(kind="text", content=text, meta={"stub": True})