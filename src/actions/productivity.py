# src/actions/productivity.py
from artifacts import Action, Artifact, Context

class PRODUCTIVITY(Action):
    name = "PRODUCTIVITY"
    requires_tools = ["synthesizer"]

    async def run(self, ctx: Context, toolset) -> Artifact:
        text = await toolset.synthesizer.synthesize("TASK_PRODUCTIVITY", None, None, ctx.query)
        return Artifact(kind="text", content=text)