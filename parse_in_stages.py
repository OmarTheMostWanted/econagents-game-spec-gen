import os
import json
import threading
import time
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from jinja2 import Template
from dotenv import load_dotenv
from econagents.llm.openai import ChatOpenAI
from logger_utils import get_logger

# --- Robust Data Classes ---
class Stage(Enum):
    META_ROLES_PHASES = "meta_roles_phases"
    STATE = "state"
    PROMPTS = "prompts"
    SETTINGS_UI = "settings_ui"

class ParserState(Enum):
    IDLE = auto()
    SELECTING_GAME = auto()
    WAITING_RESPONSE = auto()
    PROCESSING_RESPONSE = auto()
    READY_FOR_FEEDBACK = auto()
    SUCCESS = auto()
    ERROR = auto()
    WRITING_FILE = auto()

class Meta:
    def __init__(self, game_name, game_description, game_version, author1, author2, creation_date):
        self.game_name = game_name
        self.game_description = game_description
        self.game_version = game_version
        self.author1 = author1
        self.author2 = author2
        self.creation_date = creation_date

class Role:
    def __init__(self, id, name, llm, notes, phases):
        self.id = id
        self.name = name
        self.llm = llm
        self.notes = notes
        self.phases = phases

class Phase:
    def __init__(self, phase, phase_number, actionable, role_tasks):
        self.phase = phase
        self.phase_number = phase_number
        self.actionable = actionable
        self.role_tasks = role_tasks

class PayoffConsequence:
    def __init__(self, phase, role, choice, payoff):
        self.phase = phase
        self.role = role
        self.choice = choice
        self.payoff = payoff

class GameSpec:
    def __init__(self):
        self.meta: Optional[Meta] = None
        self.roles: List[Role] = []
        self.phases: List[Phase] = []
        self.payoff_consequences: List[PayoffConsequence] = []
        self.state: Optional[Dict[str, Any]] = None
        self.prompts: Optional[Dict[str, Any]] = None
        self.settings: Optional[Dict[str, Any]] = None
        self.ui: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "meta": self.meta.__dict__ if self.meta else None,
            "roles": [r.__dict__ for r in self.roles],
            "phases": [p.__dict__ for p in self.phases],
            "payoff_consequences": [pc.__dict__ for pc in self.payoff_consequences],
            "state": self.state,
            "prompts": self.prompts,
            "settings": self.settings,
            "ui": self.ui,
        }

# --- Parser Class ---
class StagedGameSpecParser:
    def __init__(self, game_spec_dir="example", prompt_dir="prompts", logger=None):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.game_spec_dir = game_spec_dir
        self.prompt_dir = prompt_dir
        self.state = ParserState.IDLE
        self.current_stage_idx = 0
        self.stages = [Stage.META_ROLES_PHASES, Stage.STATE, Stage.PROMPTS, Stage.SETTINGS_UI]
        self.stage_results: Dict[Stage, Any] = {stage: None for stage in self.stages}
        self.stage_errors: Dict[Stage, Optional[str]] = {stage: None for stage in self.stages}
        self.lock = threading.Lock()
        self.selected_game_path: Optional[str] = None
        self.game_spec = GameSpec()
        self.logger = logger or get_logger("parser")
        self.last_prompt = None
        self.last_llm_response = None
        self.logger.debug("Initialized StagedGameSpecParser")

    def list_game_specs(self) -> List[str]:
        return [os.path.join(self.game_spec_dir, f) for f in os.listdir(self.game_spec_dir)
                if os.path.isfile(os.path.join(self.game_spec_dir, f))]

    def select_game_spec(self, path: str):
        with self.lock:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Game spec not found: {path}")
            self.selected_game_path = path
            self.state = ParserState.SELECTING_GAME
            self.current_stage_idx = 0
            self.stage_results = {stage: None for stage in self.stages}
            self.stage_errors = {stage: None for stage in self.stages}
            self.game_spec = GameSpec()

    def _get_prompt_template(self, stage: Stage) -> str:
        prompt_map = {
            Stage.META_ROLES_PHASES: "meta_roles_phases_prompt.jinja2",
            Stage.STATE: "state_prompt.jinja2",
            Stage.PROMPTS: "prompts_prompt.jinja2",
            Stage.SETTINGS_UI: "settings_ui_prompt.jinja2"
        }
        template_path = os.path.join(self.prompt_dir, prompt_map[stage])
        with open(template_path, "r") as f:
            return f.read()

    def _get_instructions(self) -> str:
        with open(self.selected_game_path, "r") as f:
            return f.read()

    def _compose_context_for_stage(self, stage: Stage) -> str:
        """
        Compose context for the current stage, including only relevant data from previous stages.
        """
        context_sections = []
        if stage == Stage.STATE:
            # Pass roles, phases, payoff_consequences
            meta = self.stage_results.get(Stage.META_ROLES_PHASES, {})
            if meta:
                context_sections.append("ROLES:\n" + json.dumps(meta.get("roles", []), indent=2))
                context_sections.append("PHASES:\n" + json.dumps(meta.get("phases", []), indent=2))
                context_sections.append("PAYOFF CONSEQUENCES:\n" + json.dumps(meta.get("payoff_consequences", []), indent=2))
        elif stage == Stage.PROMPTS:
            # Pass roles, phases, tasks, state variables
            meta = self.stage_results.get(Stage.META_ROLES_PHASES, {})
            state = self.stage_results.get(Stage.STATE, {})
            if meta:
                context_sections.append("ROLES:\n" + json.dumps(meta.get("roles", []), indent=2))
                context_sections.append("PHASES:\n" + json.dumps(meta.get("phases", []), indent=2))
            if state:
                context_sections.append("STATE VARIABLES:\n" + json.dumps(state.get("state", {}), indent=2))
        elif stage == Stage.SETTINGS_UI:
            # Pass meta, roles, phases, prompts
            meta = self.stage_results.get(Stage.META_ROLES_PHASES, {})
            prompts = self.stage_results.get(Stage.PROMPTS, {})
            if meta:
                context_sections.append("META:\n" + json.dumps(meta.get("meta", {}), indent=2))
                context_sections.append("ROLES:\n" + json.dumps(meta.get("roles", []), indent=2))
                context_sections.append("PHASES:\n" + json.dumps(meta.get("phases", []), indent=2))
            if prompts:
                context_sections.append("PROMPTS:\n" + json.dumps(prompts.get("prompts", {}), indent=2))
        # Join all context sections
        return "\n--- CONTEXT ---\n" + "\n\n".join(context_sections) if context_sections else ""

    def _render_prompt(self, stage: Stage, context: Optional[str] = None) -> str:
        self.logger.debug(f"Rendering prompt for stage: {stage}")
        template_str = self._get_prompt_template(stage)
        tpl = Template(template_str)
        instructions = self._get_instructions()
        header = f"You are parsing stage: {stage.value}."
        schema_section = ""
        prompt = tpl.render(instructions=instructions, context=context or "", header=header, schema=schema_section)
        full_prompt = f"{header}\n{context or ''}\n{prompt}"
        self.last_prompt = full_prompt
        self.logger.info(f"Prompt for stage {stage.value}:\n{full_prompt}")
        return full_prompt

    async def _run_llm_async(self, prompt: str) -> str:
        self.logger.info(f"Sending prompt to LLM:\n{prompt}")
        messages = [
            {"role": "system", "content": "You are a JSON extractor."},
            {"role": "user", "content": prompt}
        ]
        tracing_extra = {}
        response = await self.llm.get_response(messages, tracing_extra)
        self.last_llm_response = response
        self.logger.info(f"LLM response:\n{response}")
        return response

    def run_stage(self, feedback: Optional[str] = None):
        with self.lock:
            stage = self.stages[self.current_stage_idx]
            self.state = ParserState.WAITING_RESPONSE
            context = self._compose_context_for_stage(stage)
            prompt = feedback if feedback else self._render_prompt(stage, context)
            # Run LLM in a thread
            thread = threading.Thread(target=self._run_stage_thread, args=(stage, prompt))
            thread.start()
            return stage.value

    def _run_stage_thread(self, stage: Stage, prompt: str):
        try:
            import asyncio
            self.logger.debug(f"Starting LLM thread for stage: {stage}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self._run_llm_async(prompt))
            self.state = ParserState.PROCESSING_RESPONSE
            self.logger.debug(f"Received LLM response for stage: {stage}")
            self._process_stage_response(stage, response)
            loop.close()
        except Exception as e:
            self.logger.error(f"Exception in LLM thread for stage {stage}: {e}")
            self.stage_errors[stage] = str(e)
            self.state = ParserState.ERROR

    def _process_stage_response(self, stage: Stage, response: str):
        self.logger.debug(f"Processing LLM response for stage: {stage}")
        try:
            data = json.loads(response)
        except Exception as e:
            self.logger.error(f"JSON decode error for stage {stage}: {e}\nRaw response:\n{response}")
            self.stage_errors[stage] = f"Invalid JSON: {e}\nRaw response:\n{response}"
            self.state = ParserState.ERROR
            return
        valid, error = self._validate_stage(stage, data)
        if not valid:
            self.logger.error(f"Validation error for stage {stage}: {error}")
            self.stage_errors[stage] = error
            self.state = ParserState.ERROR
            return
        self.stage_results[stage] = data
        self.stage_errors[stage] = None
        self._update_game_spec(stage, data)
        self.state = ParserState.SUCCESS
        self.logger.debug(f"Stage {stage} processed successfully.")

    def _validate_stage(self, stage: Stage, data: Any) -> (bool, Optional[str]):
        # Basic schema checks
        if stage == Stage.META_ROLES_PHASES:
            required = ["meta", "roles", "phases", "payoff_consequences"]
            missing = [k for k in required if k not in data]
            if missing:
                return False, f"Missing required fields: {', '.join(missing)}"
        elif stage == Stage.STATE:
            if "state" not in data:
                return False, "Missing 'state' field"
        elif stage == Stage.PROMPTS:
            if "prompts" not in data:
                return False, "Missing 'prompts' field"
        elif stage == Stage.SETTINGS_UI:
            required = ["settings", "ui"]
            missing = [k for k in required if k not in data]
            if missing:
                return False, f"Missing required fields: {', '.join(missing)}"
        return True, None

    def _update_game_spec(self, stage: Stage, data: Any):
        if stage == Stage.META_ROLES_PHASES:
            meta = data["meta"]
            self.game_spec.meta = Meta(**meta)
            self.game_spec.roles = [Role(**r) for r in data["roles"]]
            self.game_spec.phases = [Phase(**p) for p in data["phases"]]
            self.game_spec.payoff_consequences = [PayoffConsequence(**pc) for pc in data["payoff_consequences"]]
        elif stage == Stage.STATE:
            self.game_spec.state = data["state"]
        elif stage == Stage.PROMPTS:
            self.game_spec.prompts = data["prompts"]
        elif stage == Stage.SETTINGS_UI:
            self.game_spec.settings = data["settings"]
            self.game_spec.ui = data["ui"]

    def get_current_stage(self) -> str:
        return self.stages[self.current_stage_idx].value

    def get_stage_result(self) -> Any:
        return self.stage_results[self.stages[self.current_stage_idx]]

    def get_stage_error(self) -> Optional[str]:
        return self.stage_errors[self.stages[self.current_stage_idx]]

    def give_feedback(self, feedback: str):
        self.run_stage(feedback=feedback)

    def next_stage(self) -> Optional[str]:
        with self.lock:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.state = ParserState.IDLE
                return self.stages[self.current_stage_idx].value
            else:
                self.state = ParserState.SUCCESS
                return None

    def all_stages_successful(self) -> bool:
        return all(self.stage_results[s] for s in self.stages)

    def write_results_to_file(self, output_path=None):
        import datetime
        if not self.all_stages_successful():
            raise Exception("Not all stages completed successfully.")
        # Generate output file name if not provided
        if output_path is None:
            base = os.path.splitext(os.path.basename(self.selected_game_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("output", f"{base}_{timestamp}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.game_spec.to_dict(), f, indent=2)
        self.state = ParserState.WRITING_FILE
        return output_path

    def get_state(self) -> str:
        return self.state.name

    def wait_for_llm(self, poll_interval=0.5):
        while self.state == ParserState.WAITING_RESPONSE:
            time.sleep(poll_interval)

def main():
    logger = get_logger("cli")
    parser = StagedGameSpecParser(logger=logger)
    logger.debug("Started CLI main function.")
    print("\n=== EconAgents Game Spec Staged Parser ===\n")
    specs = parser.list_game_specs()
    if not specs:
        print("No game specs found in the example directory.")
        logger.error("No game specs found.")
        return
    print("Available game specs:")
    for idx, spec in enumerate(specs):
        print(f"  [{idx}] {spec}")
    while True:
        try:
            choice = int(input(f"Select a game spec [0-{len(specs)-1}]: "))
            if 0 <= choice < len(specs):
                break
            else:
                print("Invalid choice. Try again.")
        except Exception:
            print("Invalid input. Enter a number.")
    parser.select_game_spec(specs[choice])
    logger.info(f"Selected spec: {specs[choice]}")
    print(f"Selected spec: {specs[choice]}")
    print("\nStarting staged parsing...")
    while True:
        stage = parser.get_current_stage()
        print(f"\n--- Running stage: {stage} ---")
        logger.info(f"Running stage: {stage}")
        prompt = parser._render_prompt(parser.stages[parser.current_stage_idx], parser._compose_context_for_stage(parser.stages[parser.current_stage_idx]))
        print(f"\nPrompt for stage {stage}:\n{'-'*40}\n{prompt}\n{'-'*40}")
        logger.info(f"Prompt for stage {stage}:\n{prompt}")
        parser.run_stage()
        parser.wait_for_llm()
        result = parser.get_stage_result()
        error = parser.get_stage_error()
        if error:
            print(f"Error in stage {stage}: {error}")
            logger.error(f"Error in stage {stage}: {error}")
            feedback = input("Enter feedback to retry, or press Enter to continue: ")
            if feedback.strip():
                parser.give_feedback(feedback)
                parser.wait_for_llm()
                continue
        else:
            print(f"Stage {stage} result:")
            print(json.dumps(result, indent=2))
            logger.info(f"Stage {stage} result: {json.dumps(result, indent=2)}")
        next_stage = parser.next_stage()
        if not next_stage:
            break
    if parser.all_stages_successful():
        out_path = parser.write_results_to_file()
        print(f"\nAll stages complete. Results written to {out_path}")
        logger.info(f"All stages complete. Results written to {out_path}")
    else:
        print("\nNot all stages completed successfully.")
        logger.error("Not all stages completed successfully.")

if __name__ == "__main__":
    main()
