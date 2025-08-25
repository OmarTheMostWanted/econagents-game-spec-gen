
import os
import json
import asyncio
from jinja2 import Template
from econagents.llm.openai import ChatOpenAI
from dataclasses import asdict
from gamedataclasses import RolePhaseMatrix, RolePhaseTasks
from dotenv import load_dotenv

# 1. Load and chunk your instructions (text extracted from the PDF)


# Load environment variables from .env
load_dotenv()

INSTRUCTIONS_PATH = "example/Harberger.txt"
PROMPT_TEMPLATE_PATH = "prompts/role_phase_prompt.jinja2"

with open(INSTRUCTIONS_PATH, "r") as f:
    instructions = f.read()

# 2. Assemble prompt from our template


# Load prompt template from prompts directory
with open(PROMPT_TEMPLATE_PATH, "r") as f:
    prompt_tpl = Template(f.read())
prompt = prompt_tpl.render(instructions=instructions)

# 3. Instantiate EconAgents’ LLM wrapper

API_KEY = os.getenv("OPENAI_API_KEY")

# Instantiate EconAgents LLM wrapper
llm = ChatOpenAI(
    api_key=API_KEY
)

# 4. Call the model

# Async main function
async def main():
    messages = [
        {"role": "system", "content": "You are a JSON extractor."},
        {"role": "user", "content": prompt}
    ]
    tracing_extra = {}
    content = await llm.get_response(messages, tracing_extra)

    # Parse JSON from response
    matrix_json = json.loads(content)
    matrix = RolePhaseMatrix(
        roles=[
            RolePhaseTasks(role=r["role"], phase_tasks=r["phase_tasks"])
            for r in matrix_json["roles"]
        ]
    )
    print(json.dumps(asdict(matrix), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
