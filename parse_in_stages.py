import os
import json
import threading
import time
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple
from jinja2 import Template
from dotenv import load_dotenv
from econagents.llm.openai import ChatOpenAI


# --- Robust Data Classes ---
class Stage(Enum):
    META_ROLES_PHASES = "meta_roles_phases"
    STATE = "state"
    SETTINGS_UI = "settings_ui"  # moved earlier
    PARTIAL_PROMPTS = "partial_prompts"  # now final stage


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
        self.partial_prompts: Optional[List[Dict[str, str]]] = None
        self.settings: Optional[Dict[str, Any]] = None
        self.ui: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "meta": self.meta.__dict__ if self.meta else None,
            "roles": [r.__dict__ for r in self.roles],
            "phases": [p.__dict__ for p in self.phases],
            "payoff_consequences": [pc.__dict__ for pc in self.payoff_consequences],
            "state": self.state,
            "prompt_partials": self.partial_prompts,
            "settings": self.settings,
            "ui": self.ui,
        }


# --- Parser Class ---
class StagedGameSpecParser:
    def __init__(self, game_spec_dir="game_spec", prompt_dir="prompts/parsing", output_dir="output/parse_out"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.game_spec_dir = game_spec_dir
        self.prompt_dir = prompt_dir
        self.state = ParserState.IDLE
        self.current_stage_idx = 0
        self.stages = [Stage.META_ROLES_PHASES, Stage.STATE, Stage.SETTINGS_UI, Stage.PARTIAL_PROMPTS]
        self.stage_results: Dict[Stage, Any] = {stage: None for stage in self.stages}
        self.stage_errors: Dict[Stage, Optional[str]] = {stage: None for stage in self.stages}
        self.lock = threading.Lock()
        self.selected_game_path: Optional[str] = None
        self.game_spec = GameSpec()
        self.last_prompt = None
        self.last_llm_response = None
        self.output_path = output_dir
        self.expected_partial_names: List[str] = []  # track required partial names for validation

    def list_game_specs(self) -> List[str]:
        """
        List all available game spec files in the game_spec_dir.

        Returns:
            List[str]: List of file paths to game spec files.
        """
        return [os.path.join(self.game_spec_dir, f) for f in os.listdir(self.game_spec_dir)
                if os.path.isfile(os.path.join(self.game_spec_dir, f))]

    def select_game_spec(self, path: str):
        """
        Select a specific game spec file to parse and reset parser state.

        Args:
            path (str): Path to the game spec file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
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
            Stage.SETTINGS_UI: "settings_ui_prompt.jinja2",
            Stage.PARTIAL_PROMPTS: "partial_prompts_prompt.jinja2",
        }
        template_path = os.path.join(self.prompt_dir, prompt_map[stage])
        with open(template_path, "r") as f:
            return f.read()

    def _get_instructions(self) -> str:
        with open(self.selected_game_path, "r") as f:
            return f.read()

    def _compose_context_for_stage(self, stage: Stage) -> str:
        """
        Compose context for the current stage, including relevant data from previous stages.
        """
        context_sections = []
        if stage == Stage.STATE:
            meta = self.stage_results.get(Stage.META_ROLES_PHASES, {})
            if meta:
                context_sections.append("ROLES:\n" + json.dumps(meta.get("roles", []), indent=2))
                context_sections.append("PHASES:\n" + json.dumps(meta.get("phases", []), indent=2))
                context_sections.append("PAYOFF CONSEQUENCES:\n" + json.dumps(meta.get("payoff_consequences", []), indent=2))
        elif stage == Stage.SETTINGS_UI:
            meta = self.stage_results.get(Stage.META_ROLES_PHASES, {})
            state = self.stage_results.get(Stage.STATE, {})
            if meta:
                context_sections.append("META:\n" + json.dumps(meta.get("meta", {}), indent=2))
                context_sections.append("ROLES:\n" + json.dumps(meta.get("roles", []), indent=2))
                context_sections.append("PHASES:\n" + json.dumps(meta.get("phases", []), indent=2))
            if state:
                context_sections.append("STATE VARIABLES:\n" + json.dumps(state.get("state", {}), indent=2))
        elif stage == Stage.PARTIAL_PROMPTS:
            meta_stage = self.stage_results.get(Stage.META_ROLES_PHASES, {})
            state_stage = self.stage_results.get(Stage.STATE, {})
            settings_stage = self.stage_results.get(Stage.SETTINGS_UI, {})
            roles = meta_stage.get("roles", []) if meta_stage else []
            phases = meta_stage.get("phases", []) if meta_stage else []
            payoff = meta_stage.get("payoff_consequences", []) if meta_stage else []
            context_sections.append("ROLES:\n" + json.dumps(roles, indent=2))
            context_sections.append("PHASES:\n" + json.dumps(phases, indent=2))
            context_sections.append("PAYOFF CONSEQUENCES:\n" + json.dumps(payoff, indent=2))
            if state_stage:
                context_sections.append("STATE VARIABLES:\n" + json.dumps(state_stage.get("state", {}), indent=2))
            if settings_stage:
                context_sections.append("SETTINGS:\n" + json.dumps(settings_stage.get("settings", {}), indent=2))
            # Build skeleton and record expected names
            skeleton = []
            self.expected_partial_names = []
            def add_partial(name: str, **extra):
                skeleton.append({"name": name, "content": "", **extra})
                self.expected_partial_names.append(name)
            add_partial("game_description")
            add_partial("game_information")
            add_partial("game_history")  # added generic history partial
            def to_snake(name: str) -> str:
                return name.lower().replace(" ", "_")
            for ph in phases:
                if not ph.get("actionable"):
                    continue
                phase_number = ph.get("phase_number")
                phase_name = ph.get("phase")
                role_tasks = ph.get("role_tasks", {}) or {}
                for role_name, tasks in role_tasks.items():
                    if not tasks:
                        continue
                    role_snake = to_snake(role_name)
                    system_name = f"system_{role_snake}_{phase_number}"
                    user_name = f"user_{role_snake}_{phase_number}"
                    add_partial(system_name, phase=phase_name, phase_number=phase_number, role=role_name, tasks=tasks)
                    add_partial(user_name, phase=phase_name, phase_number=phase_number, role=role_name, tasks=tasks)
            context_sections.append("SKELETON:\n" + json.dumps(skeleton, indent=2))
        return "\n--- CONTEXT ---\n" + "\n\n".join(context_sections) if context_sections else ""

    def _render_prompt(self, stage: Stage, context: Optional[str] = None , include_game_spec = True) -> str:
        template_str = self._get_prompt_template(stage)
        tpl = Template(template_str)
        instructions = self._get_instructions()
        header = f"You are parsing stage: {stage.value}."
        schema_section = ""
        prompt = tpl.render(instructions=instructions, context=context or "", header=header, schema=schema_section)
        full_prompt = f"{header}\n{context or ''}\n{prompt}"
        self.last_prompt = full_prompt
        if include_game_spec:
            return full_prompt
        else:
            # recreate without instructions but without updating state
            dry_instructions = "[Game instructions omitted (in this view only)]"
            dry_prompt = tpl.render(instructions=dry_instructions, context=context or "", header=header, schema=schema_section)
            return f"{header}\n{context or ''}\n{dry_prompt}"


    async def _run_llm_async(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a JSON extractor."},
            {"role": "user", "content": prompt}
        ]
        tracing_extra = {}
        response = await self.llm.get_response(messages, tracing_extra)
        self.last_llm_response = response
        return response

    def run_stage(self, feedback: Optional[str] = None):
        """
        Run the LLM for the current stage, optionally with a custom prompt or feedback.

        Args:
            feedback (Optional[str]): Custom prompt to use for the LLM (e.g., with human feedback or previous LLM response).

        Returns:
            str: Name of the stage being run.
        """
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
        import asyncio
        def ignore_event_loop_closed(loop, context):
            exception = context.get('exception')
            if isinstance(exception, RuntimeError) and str(
                    exception) == 'Event loop is closed':
                return  # Suppress this error
            loop.default_exception_handler(context)

        loop = None  # ensure defined for finally
        try:
            loop = asyncio.new_event_loop()
            loop.set_exception_handler(ignore_event_loop_closed)
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self._run_llm_async(prompt))
            self.state = ParserState.PROCESSING_RESPONSE
            self._process_stage_response(stage, response)
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as e:
            self.stage_errors[stage] = str(e)
            self.state = ParserState.ERROR
        finally:
            if loop is not None and not loop.is_closed():
                loop.close()

    def _process_stage_response(self, stage: Stage, response: str):
        try:
            data = json.loads(response)
        except Exception as e:
            self.stage_errors[stage] = f"Invalid JSON: {e}\nRaw response:\n{response}"
            self.state = ParserState.ERROR
            return
        valid, error = self._validate_stage(stage, data)
        if not valid:
            self.stage_errors[stage] = error
            self.state = ParserState.ERROR
            return
        self.stage_results[stage] = data
        self.stage_errors[stage] = None
        self._update_game_spec(stage, data)
        self.state = ParserState.SUCCESS

    def _validate_stage(self, stage: Stage, data: Any) -> Tuple[bool, Optional[str]]:  # corrected type hint
        if stage == Stage.META_ROLES_PHASES:
            required = ["meta", "roles", "phases", "payoff_consequences"]
            missing = [k for k in required if k not in data]
            if missing:
                return False, f"Missing required fields: {', '.join(missing)}"
        elif stage == Stage.STATE:
            if "state" not in data:
                return False, "Missing 'state' field"
        elif stage == Stage.SETTINGS_UI:
            required = ["settings", "ui"]
            missing = [k for k in required if k not in data]
            if missing:
                return False, f"Missing required fields: {', '.join(missing)}"
        elif stage == Stage.PARTIAL_PROMPTS:
            if "prompt_partials" not in data:
                return False, "Missing 'prompt_partials' field"
            if not isinstance(data["prompt_partials"], list):
                return False, "'prompt_partials' must be a list"
            names = set()
            for idx, item in enumerate(data["prompt_partials"]):
                if not isinstance(item, dict) or "name" not in item or "content" not in item:
                    return False, f"Each partial must be an object with 'name' and 'content' (error at index {idx})"
                if item["name"] in names:
                    return False, f"Duplicate partial name detected: {item['name']}"
                names.add(item["name"])
            missing_required = [n for n in self.expected_partial_names if n not in names]
            if missing_required:
                return False, f"Missing required partial(s): {', '.join(missing_required)}"
            extra = [n for n in names if n not in self.expected_partial_names]
            if extra:
                return False, f"Unexpected extra partial name(s): {', '.join(extra)}"
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
        elif stage == Stage.SETTINGS_UI:
            self.game_spec.settings = data["settings"]
            self.game_spec.ui = data["ui"]
        elif stage == Stage.PARTIAL_PROMPTS:
            self.game_spec.partial_prompts = data["prompt_partials"]

    def get_current_stage(self) -> str:
        """
        Get the name of the current parsing stage.

        Returns:
            str: Name of the current stage (e.g., "meta_roles_phases").
        """
        return self.stages[self.current_stage_idx].value

    def get_stage_result(self) -> Any:
        """
        Get the parsed result of the current stage.

        Returns:
            Any: Parsed data for the current stage, or None if not available.
        """
        return self.stage_results[self.stages[self.current_stage_idx]]

    def get_stage_error(self) -> Optional[str]:
        """
        Get the error (if any) for the current stage.

        Returns:
            Optional[str]: Error message for the current stage, or None if no error.
        """
        return self.stage_errors[self.stages[self.current_stage_idx]]

    def give_feedback(self, feedback: str):
        """
        Retry the current stage with human feedback.

        Args:
            feedback (str): Feedback to include in the prompt (should also include previous LLM response for context).
        """
        self.run_stage(feedback=feedback)

    def next_stage(self) -> Optional[str]:
        """
        Advance to the next parsing stage.

        Returns:
            Optional[str]: Name of the next stage, or None if all stages are complete.
        """
        with self.lock:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.state = ParserState.IDLE
                return self.stages[self.current_stage_idx].value
            else:
                self.state = ParserState.SUCCESS
                return None

    def all_stages_successful(self) -> bool:
        """
        Check if all parsing stages completed successfully.

        Returns:
            bool: True if all stages are successful, False otherwise.
        """
        return all(self.stage_results[s] for s in self.stages)

    def write_results_to_file(self, output_path=None) -> str:
        """
        Write the final parsed game spec to a JSON file.

        Args:
            output_path (Optional[str]): Path for the output file. If not provided, a timestamped file is created in the output/ directory.

        Returns:
            str: Path to the output file.

        Raises:
            Exception: If not all stages are successful.
        """
        import datetime
        if not self.all_stages_successful():
            raise Exception("Not all stages completed successfully.")
        # Generate output file name if not provided
        if output_path is None:
            base = os.path.splitext(os.path.basename(self.selected_game_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_path, f"{base}_{timestamp}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.game_spec.to_dict(), f, indent=2)
        self.state = ParserState.WRITING_FILE
        return output_path

    def get_state(self) -> str:
        """
        Get the current parser state.

        Returns:
            str: Name of the current parser state (e.g., "IDLE", "ERROR", "SUCCESS").
        """
        return self.state.name

    def wait_for_llm(self, poll_interval=0.5):
        """
        Block until the LLM response for the current stage is ready.

        Args:
            poll_interval (float): Time in seconds between status checks.
        """
        count = 0
        while self.state == ParserState.WAITING_RESPONSE:
            if count % 10 == 0:
                print(f"Waiting for LLM response... (state: {self.state.name}), elapsed: {count * poll_interval:.1f}s")
            count += 1
            time.sleep(poll_interval)

    def _create_retry_with_feedback_prompt(self, human_feedback: Optional[str] = None) -> str:
        """
        Create a retry prompt for the current stage, including:
        - The standard prompt for the current stage
        - The previous LLM response
        - Any error message (if present)
        - Human feedback (if provided)

        Args:
            human_feedback (Optional[str]): Additional feedback from the human verifier.

        Returns:
            str: The constructed prompt for retrying the current stage.
        """
        stage = self.stages[self.current_stage_idx]
        context = self._compose_context_for_stage(stage)
        header = f"You are parsing stage: {stage.value}."
        base_prompt = self._render_prompt(stage, context)
        previous_response = self.last_llm_response or ""
        error_message = self.stage_errors.get(stage) or ""
        prompt_sections = [header, context]
        if previous_response:
            prompt_sections.append(f"\n--- PREVIOUS LLM RESPONSE ---\n{previous_response}")
        if error_message:
            prompt_sections.append(f"\n--- ERROR MESSAGE ---\n{error_message}")
        if human_feedback:
            prompt_sections.append(f"\n--- HUMAN FEEDBACK ---\n{human_feedback}")
        prompt_sections.append(f"\n--- STANDARD PROMPT ---\n{base_prompt}")
        return "\n".join(prompt_sections)

    def retry_stage_with_feedback(self, human_feedback: Optional[str] = None):
        """
        Retry the current stage with the constructed prompt including previous response, error, and human feedback.

        Args:
            human_feedback (Optional[str]): Additional feedback from the human verifier.
        """
        prompt = self._create_retry_with_feedback_prompt(human_feedback)
        self.run_stage(feedback=prompt)

    def print_next_prompt_excluding_game_instructions(self):
        """
        Print the next prompt to be sent to the LLM, excluding the game instructions.
        """
        # print in gray:
        prompt = self._render_prompt(self.stages[self.current_stage_idx], include_game_spec=False)
        print(f"\033[1;90m{prompt}\033[0m")

def main():
    parser = StagedGameSpecParser()
    print("\n=== EconAgents Game Spec Staged Parser ===\n")
    specs = parser.list_game_specs()
    if not specs:
        print("No game specs found in the example directory.")
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

    # choice = 3 # auto-select for now
    parser.select_game_spec(specs[choice])
    print(f"Selected spec: {specs[choice]}")
    print("\nStarting staged parsing...")
    while True:
        stage = parser.get_current_stage()
        print(f"\033[1;34m\n--- Running stage: {stage} ---\033[0m")
        print("\nNext prompt to be sent to LLM (excluding game instructions):")
        print('' + '+'*40)
        parser.print_next_prompt_excluding_game_instructions()
        print('' + '+'*40)

        human_satisfied = False
        human_feedback = None
        no_error = False
        parser.run_stage()
        parser.wait_for_llm()
        # Keep running the stage until it produces no errors, then ask for human feedback, if human gives feedback try again, otherwise move to next stage
        while not (human_satisfied or no_error):
            human_satisfied = False
            state = parser.get_state()
            if state == "ERROR":
                error = parser.get_stage_error()
                print(f"\033[1;31mError in stage {stage}:\033[0m {error}")
                print("Sending retry prompt")
                parser.retry_stage_with_feedback(human_feedback)
                parser.wait_for_llm()
                continue
            elif state == "SUCCESS":
                no_error = True
                result = parser.get_stage_result()
                assert parser.get_stage_error() is None # sanity check
                print(f"\033[1;32mStage {stage} completed successfully.\033[0m")
                print(f"Parsed result:\n{json.dumps(result, indent=2)}")
                while True:
                    # feedback = input("Are you satisfied with this result? (y/n): ").strip().lower()
                    feedback = 'y'  # auto-approve for now
                    if feedback in ['y', 'n']:
                        human_satisfied = (feedback == 'y')
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                if not human_satisfied:
                    human_feedback = input("Please provide your feedback for retrying the stage: ").strip()
                    print("Sending retry prompt with human feedback...")
                    parser.retry_stage_with_feedback(human_feedback)
                    parser.wait_for_llm()
                    continue
                else:
                    next_stage = parser.next_stage()
                    if next_stage:
                        # blue
                        print(f"\033[1;34m\n--- Moving to next stage: {next_stage} ---\033[0m")
                        human_satisfied = False
                        no_error = False
                        print('' + '+'*40)
                        parser.print_next_prompt_excluding_game_instructions()
                        print('' + '+'*40)
                        parser.run_stage()
                        parser.wait_for_llm()
                    else:
                        # Bold lime green
                        print(f"\033[1;92mAll stages completed successfully!\033[0m")
                        output_path = parser.write_results_to_file()
                        print(f"Final game spec written to: {output_path}")
                        return

if __name__ == "__main__":
    main()
