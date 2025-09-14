import datetime
import os
import json
import threading
import time
from enum import Enum, auto
from typing import Optional
from jinja2 import Template
from dotenv import load_dotenv
from econagents.llm.openai import ChatOpenAI

class InterpreterState(Enum):
    IDLE = auto()
    SELECTING_JSON_SPEC = auto()
    WAITING_RESPONSE = auto()
    PROCESSING_RESPONSE = auto()
    READY_FOR_FEEDBACK = auto()
    SUCCESS = auto()
    ERROR = auto()
    WRITING_FILE = auto()

class YamlInterpreter:
    def __init__(self, prompt_dir="prompts/interpreting", yaml_template_file = 'templates/econagents_template.yaml.jinja2' ,json_spec_dir="output/parse_out" , yaml_output_dir="output/interpret_out" ):
        self.yaml_output_dir = yaml_output_dir
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.prompt_dir = prompt_dir
        self.state = InterpreterState.IDLE
        self.json_spec = None
        self.yaml_template_file = yaml_template_file
        self.yaml_template = None
        self.last_prompt = None
        self.last_llm_response = None
        self.result_yaml = None
        self.error = None
        self.lock = threading.Lock()


    def load_json_spec(self, path: str):
        """
        Load the parsed JSON spec from file and set state to SELECTING_JSON_SPEC.
        """
        with open(path, "r") as f:
            self.json_spec = json.load(f)
        self.state = InterpreterState.SELECTING_JSON_SPEC

    def load_yaml_template(self, path: str):
        """
        Load the YAML template from file.
        """
        with open(path, "r") as f:
            self.yaml_template = f.read()

    def _get_prompt_template(self) -> str:
        template_path = os.path.join(self.prompt_dir, "interpret_yaml_prompt.jinja2")
        with open(template_path, "r") as f:
            return f.read()

    def render_interpret_prompt(self, human_feedback: Optional[str] = None, include_json_spec = True) -> str:
        """
        Render the prompt for the LLM to fill out the YAML template.
        """
        tpl = Template(self._get_prompt_template())
        header = "You are an JSON to YAML interpreter."
        if include_json_spec:
            json_spec_str = json.dumps(self.json_spec, indent=2)
        else:
            json_spec_str = "<JSON spec omitted in this view>"
        prompt = tpl.render(header=header, json_spec=json_spec_str, human_feedback=human_feedback or "", yaml_template=self.yaml_template)
        self.last_prompt = prompt
        return prompt

    async def _run_llm_async(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a YAML generator."},
            {"role": "user", "content": prompt}
        ]
        response = await self.llm.get_response(messages, {})
        self.last_llm_response = response
        return response

    def run_interpret(self, human_feedback: Optional[str] = None):
        """
        Run the LLM to fill out the YAML template.
        """
        with self.lock:
            self.state = InterpreterState.WAITING_RESPONSE
            prompt = self.render_interpret_prompt(human_feedback)
            thread = threading.Thread(target=self._run_interpret_thread, args=(prompt,))
            thread.start()

    def _run_interpret_thread(self, prompt: str):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self._run_llm_async(prompt))
            self.state = InterpreterState.PROCESSING_RESPONSE
            self._process_llm_response(response)
        except Exception as e:
            self.error = str(e)
            self.state = InterpreterState.ERROR
        finally:
            loop.close()

    def _process_llm_response(self, response: str):
        self.result_yaml = response
        self.error = None
        self.state = InterpreterState.READY_FOR_FEEDBACK

    def wait_for_llm(self, poll_interval=0.5):
        count = 0
        while self.state == InterpreterState.WAITING_RESPONSE:
            if count % 10 == 0:
                print(f"Waiting for LLM response... (elapsed {count * poll_interval:.1f}s)")
            count += 1
            time.sleep(poll_interval)

    def get_result(self) -> Optional[str]:
        return self.result_yaml

    def get_error(self) -> Optional[str]:
        return self.error

    def give_feedback(self, human_feedback: str):
        self.state = InterpreterState.READY_FOR_FEEDBACK
        self.run_interpret(human_feedback)

    def write_yaml_to_file(self, path=None) -> str:
        """
        Write the interpreted YAML to a file.
        """
        if not self.result_yaml:
            raise Exception("No YAML result to write.")
        self.state = InterpreterState.WRITING_FILE
        if path is None:
            time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.yaml_output_dir, f"interpreted_{time_stamp}.yaml")
        with open(path, "w") as f:
            f.write(self.result_yaml)
        return path

    def create_retry_with_feedback_prompt(self, human_feedback: Optional[str] = None) -> str:
        """
        Create a retry prompt including previous LLM response, error, and human feedback.
        """
        header = "You are an expert YAML interpreter."
        json_spec_str = json.dumps(self.json_spec, indent=2)
        previous_response = self.last_llm_response or ""
        error_message = self.error or ""
        tpl = Template(self._get_prompt_template())
        prompt = tpl.render(header=header, json_spec=json_spec_str, human_feedback=human_feedback or "", yaml_template=self.yaml_template)
        prompt_sections = [header, f"\nJSON Spec:\n{json_spec_str}"]
        if previous_response:
            prompt_sections.append(f"\n--- PREVIOUS LLM RESPONSE ---\n{previous_response}")
        if error_message:
            prompt_sections.append(f"\n--- ERROR MESSAGE ---\n{error_message}")
        if human_feedback:
            prompt_sections.append(f"\n--- HUMAN FEEDBACK ---\n{human_feedback}")
        prompt_sections.append(f"\n--- STANDARD PROMPT ---\n{prompt}")
        return "\n".join(prompt_sections)

# CLI for testing
if __name__ == "__main__":
    interp = YamlInterpreter()
    print("\n=== EconAgents YAML Interpreter ===\n")
    json_path = 'output/parse_out/dictator_20250913_215213.json'
    yaml_template_path = interp.yaml_template_file
    interp.load_json_spec(json_path)
    interp.load_yaml_template(yaml_template_path)
    print("\nPrompt to be sent to LLM:")
    print('-'*40)
    print(interp.render_interpret_prompt(None , include_json_spec=False))
    print('-'*40)
    interp.run_interpret()
    interp.wait_for_llm()
    result = interp.get_result()
    error = interp.get_error()
    print("\nInterpreted YAML result:")
    print(result)
    if error:
        print(f"Error: {error}")
    while True:
        feedback = input("Enter feedback to retry, or press Enter to continue: ").strip()
        if feedback:
            retry_prompt = interp.create_retry_with_feedback_prompt(feedback)
            print("\nRetry prompt:")
            print(retry_prompt)
            interp.run_interpret(feedback)
            interp.wait_for_llm()
            result = interp.get_result()
            error = interp.get_error()
            print("\nInterpreted YAML result after retry:")
            print(result)
            if error:
                print(f"Error: {error}")
        else:
            break
    save_path = input("Enter path to save YAML (or press Enter for default): ").strip()
    out_path = interp.write_yaml_to_file(save_path or None)
    print(f"YAML written to: {out_path}")

