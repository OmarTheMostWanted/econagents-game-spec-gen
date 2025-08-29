
import os
import json
import asyncio
from jinja2 import Template
from econagents.llm.openai import ChatOpenAI
from dataclasses import asdict
from gamedataclasses import PhaseRoleMatrix, PhaseRoleTasks
from dotenv import load_dotenv

# 1. Load and chunk your instructions (text extracted from the PDF)


# Load environment variables from .env
load_dotenv()

INSTRUCTIONS_PATH = "example/harberger.txt"
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

    # Parse JSON from response, robust to missing fields
    try:
        matrix_json = json.loads(content)
        phases_json = matrix_json.get("phases", [])
        matrix = PhaseRoleMatrix(
            phases=[
                PhaseRoleTasks(
                    phase=p.get("phase", None),
                    role_tasks=p.get("role_tasks", None)
                )
                for p in phases_json
            ]
        )
        # Display a human-friendly role/task matrix for each phase
        for phase in matrix.phases:
            print(f"\n=== Phase: {phase.phase} ===")
            if phase.role_tasks:
                for role, tasks in phase.role_tasks.items():
                    print(f"Role: {role}")
                    for task in tasks:
                        print(f"  - {task}")
            else:
                print("No roles/tasks found.")
            print("---------------------------")
        # Also print the raw JSON for reference
        print(json.dumps(asdict(matrix), indent=2))
    except Exception as e:
        print("Error parsing LLM response:", e)
        print("Raw response:", content)
        # Feedback prompt for reattempt
        feedback_prompt = f"""
        The previous response was incorrectly formatted or missing required fields. Please return strictly valid JSON with the following schema:
        {{
          "phases": [
            {{
              "phase": "<PhaseName>",
              "role_tasks": {{
                "<Role1>": ["<task1>", "<task2>", ...],
                ...
              }}
            }},
            ...
          ]
        }}
        The previous response was:
        {content}
        """
        print("\n--- FEEDBACK PROMPT ---\n")
        print(feedback_prompt)


def run_parser_for_gui():
        """
        Synchronous wrapper for GUI use. Returns (PhaseRoleMatrix, feedback_prompt)
        """
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_run_parser_for_gui_async())

async def _run_parser_for_gui_async():
        messages = [
                {"role": "system", "content": "You are a JSON extractor."},
                {"role": "user", "content": prompt}
        ]
        tracing_extra = {}
        content = await llm.get_response(messages, tracing_extra)

        try:
                matrix_json = json.loads(content)
                phases_json = matrix_json.get("phases", [])
                matrix = PhaseRoleMatrix(
                        phases=[
                                PhaseRoleTasks(
                                        phase=p.get("phase", None),
                                        phase_number=p.get("phase_number", None),
                                        actionable=p.get("actionable", None),
                                        role_tasks=p.get("role_tasks", None)
                                )
                                for p in phases_json
                        ]
                )
                return matrix, ""
        except Exception as e:
                feedback_prompt = f"""
                The previous response was incorrectly formatted or missing required fields. Please return strictly valid JSON with the following schema:
                {{
                    "phases": [
                        {{
                            "phase": "<PhaseName>",
                            "phase_number": <int>,
                            "actionable": <bool>,
                            "role_tasks": {{
                                "<Role1>": ["<task1>", "<task2>", ...],
                                ...
                            }}
                        }},
                        ...
                    ]
                }}
                The previous response was:
                {content}
                """
                return None, feedback_prompt

if __name__ == "__main__":
    asyncio.run(main())
