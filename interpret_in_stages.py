import os
import json
import threading
import time
from enum import Enum, auto
from typing import Dict, Any, Optional
from jinja2 import Template
from dotenv import load_dotenv
from econagents.llm.openai import ChatOpenAI

# --- Stages for YAML interpretation ---
class InterpretStage(Enum):
    META = "meta"
    ROLES = "roles"
    STATE = "state"
    PHASES = "phases"
    PROMPTS = "prompts"
    SETTINGS = "settings"
    UI = "ui"

class InterpreterState(Enum):
    IDLE = auto()
    SELECTING_SPEC = auto()
    WAITING_RESPONSE = auto()
    PROCESSING_RESPONSE = auto()
    READY_FOR_FEEDBACK = auto()
    SUCCESS = auto()
    ERROR = auto()
    WRITING_FILE = auto()

class StagedYamlInterpreter:
    def __init__(self, parsed_json_path, prompt_dir="prompts/interpreting", output_dir="output/interpret_out"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.parsed_json_path = parsed_json_path
        self.prompt_dir = prompt_dir
        self.state = InterpreterState.IDLE
        self.current_stage_idx = 0
        self.stages = [
            InterpretStage.META,
            InterpretStage.ROLES,
            InterpretStage.STATE,
            InterpretStage.PHASES,
            InterpretStage.PROMPTS,
            InterpretStage.SETTINGS,
            InterpretStage.UI
        ]
        self.stage_results: Dict[InterpretStage, Any] = {stage: None for stage in self.stages}
        self.stage_errors: Dict[InterpretStage, Optional[str]] = {stage: None for stage in self.stages}
        self.lock = threading.Lock()
        self.last_prompt = None
        self.last_llm_response = None
        self.output_path = output_dir
        self.yaml_data: Dict[str, Any] = {}  # Store all root fields here

    def _get_prompt_template(self, stage: InterpretStage) -> str:
        prompt_map = {
            InterpretStage.META: "meta_prompt.jinja2",
            InterpretStage.ROLES: "roles_prompt.jinja2",
            InterpretStage.STATE: "state_prompt.jinja2",
            InterpretStage.PHASES: "phases_prompt.jinja2",
            InterpretStage.PROMPTS: "prompts_prompt.jinja2",
            InterpretStage.SETTINGS: "settings_prompt.jinja2",
            InterpretStage.UI: "ui_prompt.jinja2"
        }
        template_path = os.path.join(self.prompt_dir, prompt_map[stage])
        with open(template_path, "r") as f:
            return f.read()

    def _get_json_data(self) -> Any:
        with open(self.parsed_json_path, "r") as f:
            return json.load(f)

    def _render_prompt(self, stage: InterpretStage, include_json_spec = True) -> str:
        template_str = self._get_prompt_template(stage)
        tpl = Template(template_str)
        json_data = self._get_json_data()
        prompt = tpl.render(json_data=json.dumps(json_data, indent=2))
        self.last_prompt = prompt
        if include_json_spec:
            return prompt
        else:
            # return prompt without the JSON spec but without updating state
            return tpl.render(json_data='[JSON data omitted in this view]')

    async def _run_llm_async(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a YAML extractor."},
            {"role": "user", "content": prompt}
        ]
        tracing_extra = {}
        response = await self.llm.get_response(messages, tracing_extra)
        self.last_llm_response = response
        return response

    def run_stage(self, feedback: Optional[str] = None):
        with self.lock:
            stage = self.stages[self.current_stage_idx]
            self.state = InterpreterState.WAITING_RESPONSE
            prompt = feedback if feedback else self._render_prompt(stage)
            thread = threading.Thread(target=self._run_stage_thread, args=(stage, prompt))
            thread.start()
            return stage.value

    def _run_stage_thread(self, stage: InterpretStage, prompt: str):
        import asyncio
        def ignore_event_loop_closed(loop, context):
            exception = context.get('exception')
            if isinstance(exception, RuntimeError) and str(exception) == 'Event loop is closed':
                return
            loop.default_exception_handler(context)
        loop = None
        try:
            loop = asyncio.new_event_loop()
            loop.set_exception_handler(ignore_event_loop_closed)
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self._run_llm_async(prompt))
            self.state = InterpreterState.PROCESSING_RESPONSE
            self._process_stage_response(stage, response)
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as e:
            self.stage_errors[stage] = str(e)
            self.state = InterpreterState.ERROR
        finally:
            if loop is not None:
                loop.close()

    def _process_stage_response(self, stage: InterpretStage, response: str):
        import yaml
        root_key = stage.value
        # Try to parse the LLM response as JSON
        try:
            parsed = json.loads(response)
        except Exception as e:
            self.stage_errors[stage] = f"Invalid JSON response: {e}"
            self.state = InterpreterState.ERROR
            return
        # If the LLM output is a dict with the root key, extract its value
        if isinstance(parsed, dict) and root_key in parsed:
            value = parsed[root_key]
        else:
            value = parsed
        self.stage_results[stage] = value
        self.stage_errors[stage] = None
        self.yaml_data[root_key] = value
        self.state = InterpreterState.SUCCESS

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
                self.state = InterpreterState.IDLE
                return self.stages[self.current_stage_idx].value
            else:
                self.state = InterpreterState.SUCCESS
                return None

    def all_stages_successful(self) -> bool:
        return all(self.stage_results[s] for s in self.stages)

    def write_results_to_file(self, output_path=None) -> str:
        import datetime
        import yaml
        if not self.all_stages_successful():
            raise Exception("Not all stages completed successfully.")
        if output_path is None:
            base = os.path.splitext(os.path.basename(self.parsed_json_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_path, f"{base}_{timestamp}.yaml")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Dump the entire yaml_data dict as YAML
        with open(output_path, "w") as f:
            yaml.dump(self.yaml_data, f, sort_keys=False, allow_unicode=True)
        self.state = InterpreterState.WRITING_FILE
        return output_path

    def get_state(self) -> str:
        return self.state.name

    def wait_for_llm(self, poll_interval=0.5):
        count = 0
        while self.state == InterpreterState.WAITING_RESPONSE:
            if count % 100 == 0:
                print(f"Waiting for LLM response... (state: {self.state.name})")
            count += 1
            time.sleep(poll_interval)

    def _create_retry_with_feedback_prompt(self, human_feedback: Optional[str] = None) -> str:
        stage = self.stages[self.current_stage_idx]
        base_prompt = self._render_prompt(stage)
        previous_response = self.last_llm_response or ""
        error_message = self.stage_errors.get(stage) or ""
        prompt_sections = [f"You are interpreting stage: {stage.value}."]
        if previous_response:
            prompt_sections.append(f"\n--- PREVIOUS LLM RESPONSE ---\n{previous_response}")
        if error_message:
            prompt_sections.append(f"\n--- ERROR MESSAGE ---\n{error_message}")
        if human_feedback:
            prompt_sections.append(f"\n--- HUMAN FEEDBACK ---\n{human_feedback}")
        prompt_sections.append(f"\n--- STANDARD PROMPT ---\n{base_prompt}")
        return "\n".join(prompt_sections)

    def retry_stage_with_feedback(self, human_feedback: Optional[str] = None):
        prompt = self._create_retry_with_feedback_prompt(human_feedback)
        self.run_stage(feedback=prompt)

    def print_next_prompt(self):
        prompt = self._render_prompt(self.stages[self.current_stage_idx] , include_json_spec=False)
        print(f"\033[1;90m{prompt}\033[0m")

def main():
    # Example usage: select a parsed JSON file from output/parse_out
    parse_dir = "output/parse_out"
    files = [os.path.join(parse_dir, f) for f in os.listdir(parse_dir) if f.endswith(".json")]
    if not files:
        print("No parsed JSON specs found.")
        return
    print("Available parsed specs:")
    for idx, spec in enumerate(files):
        print(f"  [{idx}] {spec}")
    choice = 0 # auto-select for now
    parsed_json_path = files[choice]
    print(f"Selected parsed spec: {parsed_json_path}")
    interpreter = StagedYamlInterpreter(parsed_json_path)
    print("\nStarting staged YAML interpretation...")
    while True:
        stage = interpreter.get_current_stage()
        print(f"\033[1;34m\n--- Interpreting stage: {stage} ---\033[0m")
        print("\nNext prompt to be sent to LLM:")
        print('' + '+'*40)
        interpreter.print_next_prompt()
        print('' + '+'*40)
        human_satisfied = False
        human_feedback = None
        no_error = False
        interpreter.run_stage()
        interpreter.wait_for_llm()
        while not (human_satisfied or no_error):
            human_satisfied = False
            state = interpreter.get_state()
            if state == "ERROR":
                error = interpreter.get_stage_error()
                print(f"\033[1;31mError in stage {stage}:\033[0m {error}")
                print("Sending retry prompt")
                interpreter.retry_stage_with_feedback(human_feedback)
                interpreter.wait_for_llm()
                continue
            elif state == "SUCCESS":
                no_error = True
                result = interpreter.get_stage_result()
                assert interpreter.get_stage_error() is None
                print(f"\033[1;32mStage {stage} completed successfully.\033[0m")
                print(f"YAML section:\n{result}")
                while True:
                    feedback = 'y'  # auto-approve for now
                    if feedback in ['y', 'n']:
                        human_satisfied = (feedback == 'y')
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                if not human_satisfied:
                    human_feedback = input("Please provide your feedback for retrying the stage: ").strip()
                    print("Sending retry prompt with human feedback...")
                    interpreter.retry_stage_with_feedback(human_feedback)
                    interpreter.wait_for_llm()
                    continue
                else:
                    next_stage = interpreter.next_stage()
                    if next_stage:
                        print(f"\033[1;34m\n--- Moving to next stage: {next_stage} ---\033[0m")
                        human_satisfied = False
                        no_error = False
                        print('' + '+'*40)
                        interpreter.print_next_prompt()
                        print('' + '+'*40)
                        interpreter.run_stage()
                        interpreter.wait_for_llm()
                    else:
                        print(f"\033[1;92mAll stages completed successfully!\033[0m")
                        output_path = interpreter.write_results_to_file()
                        print(f"Final YAML written to: {output_path}")
                        return

if __name__ == "__main__":
    main()
