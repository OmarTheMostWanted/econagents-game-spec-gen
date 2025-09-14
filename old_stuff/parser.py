import os
import json
import asyncio
from jinja2 import Template
from econagents.llm.openai import ChatOpenAI
from dataclasses import asdict
from gamedataclasses import PhaseRoleMatrix, PhaseRoleTasks, PayoffConsequence
from dotenv import load_dotenv
from enum import Enum, auto

class ParserState(Enum):
    IDLE = auto()
    SELECTING_GAME = auto()
    PARSING = auto()
    WAITING_RESPONSE = auto()
    SUCCESS = auto()
    ERROR = auto()
    RETRY = auto()

class GameSpecParser:
    def __init__(self, prompt_template_path="prompts/role_phase_prompt.jinja2", game_spec_dir="example"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.prompt_template_path = prompt_template_path
        self.game_specification_directory = game_spec_dir
        self.state = ParserState.IDLE
        self.last_error = None
        self.last_feedback_prompt = None
        self.result = None
        self.selected_game = None
        self.available_games = self.scan_game_specification_directory()

    def scan_game_specification_directory(self):
        # List all files in game_specification_directory (ignore directories)
        if not os.path.isdir(self.game_specification_directory):
            return []
        return [f for f in os.listdir(self.game_specification_directory)
                if os.path.isfile(os.path.join(self.game_specification_directory, f))]

    def start(self, game_filename=None):
        self.state = ParserState.SELECTING_GAME
        self.last_error = None
        self.last_feedback_prompt = None
        self.result = None
        if game_filename and game_filename in self.available_games:
            self.selected_game = game_filename
        else:
            self.selected_game = self.available_games[0] if self.available_games else None
        self.state = ParserState.PARSING
        return self.selected_game

    def get_prompt(self):
        if not self.selected_game:
            self.last_error = "No game selected."
            self.state = ParserState.ERROR
            return None
        instructions_path = os.path.join(self.game_specification_directory, self.selected_game)
        with open(instructions_path, "r") as f:
            instructions = f.read()
        with open(self.prompt_template_path, "r") as f:
            prompt_tpl = Template(f.read())
        prompt = prompt_tpl.render(instructions=instructions)
        return prompt

    async def parse(self):
        self.state = ParserState.WAITING_RESPONSE
        prompt = self.get_prompt()
        if not prompt:
            return None
        messages = [
            {"role": "system", "content": "You are a JSON extractor."},
            {"role": "user", "content": prompt}
        ]
        tracing_extra = {}
        try:
            content = await self.llm.get_response(messages, tracing_extra)
            self.handle_response(content)
        except Exception as e:
            self.last_error = str(e)
            self.state = ParserState.ERROR
            self.result = None
        return self.result

    def handle_response(self, content):
        # Step 1: Validate JSON syntax
        try:
            matrix_json = json.loads(content)
        except json.JSONDecodeError as e:
            self.last_error = f"The response could not be parsed as JSON. Error: {str(e)}"
            self.last_feedback_prompt = (
                "The model's response was not valid JSON. "
                "Please ensure the output is strictly valid JSON matching the schema.\n"
                f"Error details: {str(e)}\n"
                "Example schema:\n"
                "- phases: list of phase objects with phase, phase_number, actionable, role_tasks\n"
                "- payoff_consequences: list of payoff consequence objects with phase, role, choice, payoff\n"
                f"Raw response:\n{content}"
            )
            self.state = ParserState.ERROR
            self.result = None
            return

        # Step 2: Parse dataclasses and check for missing/null fields
        phases_json = matrix_json.get("phases", [])
        payoff_json = matrix_json.get("payoff_consequences", [])
        valid_phases = []
        missing_fields = []
        for idx, p in enumerate(phases_json):
            missing = []
            if p.get("phase", None) is None:
                missing.append("phase")
            if p.get("phase_number", None) is None:
                missing.append("phase_number")
            if p.get("actionable", None) is None:
                missing.append("actionable")
            if p.get("role_tasks", None) is None:
                missing.append("role_tasks")
            if missing:
                missing_fields.append((idx, missing))
                continue
            valid_phases.append(
                PhaseRoleTasks(
                    phase=p["phase"],
                    phase_number=int(p["phase_number"]),
                    actionable=bool(p["actionable"]),
                    role_tasks=p["role_tasks"]
                )
            )
        # Step 3: Check payoff consequences for missing/null fields
        valid_payoffs = []
        payoff_missing_fields = []
        for idx, pc in enumerate(payoff_json):
            missing = []
            for field in ("phase", "role", "choice", "payoff"):
                if pc.get(field, None) is None:
                    missing.append(field)
            if missing:
                payoff_missing_fields.append((idx, missing))
                continue
            valid_payoffs.append(
                PayoffConsequence(
                    phase=pc["phase"],
                    role=pc["role"],
                    choice=pc["choice"],
                    payoff=pc["payoff"]
                )
            )
        # Step 4: Error reporting and feedback prompt
        if missing_fields or payoff_missing_fields or not valid_phases:
            error_msgs = []
            if missing_fields:
                for idx, fields in missing_fields:
                    error_msgs.append(
                        f"Phase {idx+1} is missing required fields: {', '.join(fields)}."
                    )
            if payoff_missing_fields:
                for idx, fields in payoff_missing_fields:
                    error_msgs.append(
                        f"Payoff consequence {idx+1} is missing required fields: {', '.join(fields)}."
                    )
            if not valid_phases:
                error_msgs.append("No valid phases found in the response.")
            self.last_error = "\n".join(error_msgs)
            self.last_feedback_prompt = (
                "There were issues with the model's response:\n"
                + self.last_error + "\n"
                "Please ensure all required fields are present and not null.\n"
                "Required for each phase: phase, phase_number, actionable, role_tasks.\n"
                "Required for each payoff consequence: phase, role, choice, payoff.\n"
                "Example of a valid phase object:\n"
                "{\"phase\": \"Offer\", \"phase_number\": 1, \"actionable\": true, \"role_tasks\": {\"Trader\": [\"Make offer\"]}}\n"
                "Example of a valid payoff consequence object:\n"
                "{\"phase\": \"Offer\", \"role\": \"Trader\", \"choice\": \"Make offer\", \"payoff\": \"Trader receives payoff based on offer\"}\n"
                f"Raw response:\n{content}"
            )
            self.state = ParserState.ERROR
            self.result = None
            return
        # Step 5: Success
        matrix = PhaseRoleMatrix(
            phases=valid_phases,
            payoff_consequences=valid_payoffs
        )
        self.result = matrix
        self.state = ParserState.SUCCESS
        self.last_error = None
        self.last_feedback_prompt = None

    def get_feedback_prompt(self, raw_response, json_error=False):
        if json_error:
            reason = f"JSON parsing error: {self.last_error}"
        else:
            reason = "Missing required fields (phase_number, actionable) in one or more phases."
        prompt_lines = [
            "--- FEEDBACK ---",
            "There was a problem with the model's response:",
            f"Reason: {reason}",
            "",
            "Please ensure the output is strictly valid JSON and matches the required schema.",
            "",
            "Required for each phase:",
            "  - phase (string)",
            "  - phase_number (integer)",
            "  - actionable (boolean)",
            "  - role_tasks (dictionary of role to list of tasks)",
            "",
            "Required for each payoff consequence:",
            "  - phase (string)",
            "  - role (string)",
            "  - choice (string)",
            "  - payoff (string)",
            "",
            "Example of a valid phase object:",
            '  {"phase": "Offer", "phase_number": 1, "actionable": true, "role_tasks": {"Trader": ["Make offer"]}}',
            "",
            "Example of a valid payoff consequence object:",
            '  {"phase": "Offer", "role": "Trader", "choice": "Make offer", "payoff": "Trader receives payoff based on offer"}',
            "",
            "Here is the raw response received:",
            raw_response,
            "--- END FEEDBACK ---"
        ]
        return "\n".join(prompt_lines)

    def get_combined_feedback(self, human_feedback=None):
        """
        Returns a feedback prompt for the LLM that always includes the original prompt and instructions,
        followed by either automatic or human feedback.
        """
        original_prompt = self.get_prompt() or ""
        feedback_section = human_feedback if human_feedback is not None else self.last_feedback_prompt or ""
        prompt_lines = [
            "ORIGINAL PROMPT:",
            original_prompt,
            "",
            "FEEDBACK (please use this to improve your output):",
            feedback_section,
        ]
        return "\n".join(prompt_lines)

    def reset(self):
        self.state = ParserState.IDLE
        self.last_error = None
        self.last_feedback_prompt = None
        self.result = None
        self.selected_game = None
