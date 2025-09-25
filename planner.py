# read env file
import os
from dotenv import load_dotenv
from prompt import PLANNER_PROMPT, USER_PROMPT, SYSTEM_PROMPT_MULTI
load_dotenv()

# get env

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY_AGENT = os.getenv("OPENAI_API_KEY_AGENT")
OPENAI_API_MODEL_AGENT = os.getenv("OPENAI_API_MODEL_AGENT")
OPENAI_API_MODEL_AGENT_PLANNER = os.getenv("OPENAI_API_MODEL_AGENT_PLANNER")

# init openai client
import openai

client = openai.OpenAI(api_key=OPENAI_API_KEY_AGENT, base_url=OPENAI_API_BASE)



# init planner

USER_INPUT = "How many years passed between the start of World War I and the moon landing?"

response = client.chat.completions.create(
    model=OPENAI_API_MODEL_AGENT_PLANNER,
    messages=[
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content":  "User: " + USER_INPUT}
    ]
)

print(response.choices[0].message.content)