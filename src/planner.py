# read env file
import os
from dotenv import load_dotenv
from prompt import PLANNER_PROMPT
load_dotenv()
import time
# get time context
from utils.timectx import parse_time_intent







# get env
from config import get_config
OPENAI_API_BASE = get_config().get('external_services.openai.api_base', 'https://api.openai.com/v1')
OPENAI_API_KEY_AGENT = os.getenv("OPENAI_API_KEY_AGENT")
OPENAI_API_MODEL_AGENT = get_config().get('models.agent.model_name', 'gpt-4')
OPENAI_API_MODEL_AGENT_PLANNER = get_config().get('models.planner.model_name', 'gpt-4')

# init openai client
import openai

client = openai.OpenAI(api_key=OPENAI_API_KEY_AGENT, base_url=OPENAI_API_BASE)



# init planner

USER_INPUT = "郭毅可什么时候可以当浸会大学校长？"

time_start = time.time()
response = client.chat.completions.create(
    model=OPENAI_API_MODEL_AGENT_PLANNER,
    messages=[
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content":  "User: " + USER_INPUT}
    ]
)
time_end = time.time()  
print(response.choices[0].message.content)
print(f"Time taken: {time_end - time_start} seconds")