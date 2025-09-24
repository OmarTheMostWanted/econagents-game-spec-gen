import os
import json
import time
import threading
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
from econagents.llm.openai import ChatOpenAI
from yaml_dataclasses import (
    ExperimentConfig,
    PromptPartial,
    AgentRoleConfig,
    RolePromptEntry,
    AgentMappingConfig,
    StateFieldConfig,
    StateConfig,
    ManagerConfig,
    RunnerConfig,
)

TEMPLATE_PATH = "templates"
TEMPLATE_FILE = "econagents_template.yaml.jinja2"
PROMPTS_DIR = "prompts/interpret"
OUTPUT_DIR = "output/experiment_yaml"
PARSED_JSON_DIR = "output/parse_out"

# ---------------- Stages -----------------
class Stage(Enum):
    META_PARTIALS = "meta_partials"
    ROLES_PHASES = "roles_phases"
    STATE = "state"
    PROMPT_PARTIALS = "prompt_partials"  # new stage to request specific partials
    ROLE_PROMPTS = "role_prompts"
    AGENTS = "agents"
    FINALIZE = "finalize"

class StageState(Enum):
    IDLE = auto()
    WAITING = auto()
    PROCESSING = auto()
    SUCCESS = auto()
    ERROR = auto()

# ---------------- Interpreter --------------
class InterpretInStages:
    def __init__(self, parsed_json_path: str):
        load_dotenv()
        self.parsed_json_path = parsed_json_path
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
        self.template_env = Environment(loader=FileSystemLoader(TEMPLATE_PATH), trim_blocks=True, lstrip_blocks=True)
        self.template = self.template_env.get_template(TEMPLATE_FILE)
        self.stages: List[Stage] = [
            Stage.META_PARTIALS,
            Stage.ROLES_PHASES,
            Stage.STATE,
            Stage.PROMPT_PARTIALS,
            Stage.ROLE_PROMPTS,
            Stage.AGENTS,
            Stage.FINALIZE,
        ]
        self.current_index = 0
        self.state = StageState.IDLE
        self.results: Dict[Stage, Any] = {s: None for s in self.stages}
        self.errors: Dict[Stage, Optional[str]] = {s: None for s in self.stages}
        self.last_prompt: Optional[str] = None
        self.last_response: Optional[str] = None
        self.lock = threading.Lock()
        self.experiment: Optional[ExperimentConfig] = None

    # ------------- Helpers -------------
    def _load_parsed(self) -> Dict[str, Any]:
        with open(self.parsed_json_path, 'r') as f:
            return json.load(f)

    def _render_stage_prompt(self, stage: Stage) -> str:
        if stage == Stage.FINALIZE:
            return "No LLM prompt for finalize stage."
        parsed = self._load_parsed()
        template_map = {
            Stage.META_PARTIALS: "meta_partials_prompt.jinja2",
            Stage.ROLES_PHASES: "roles_phases_prompt.jinja2",
            Stage.STATE: "state_prompt.jinja2",
            Stage.PROMPT_PARTIALS: "prompt_partials_prompt.jinja2",
            Stage.ROLE_PROMPTS: "role_prompts_prompt.jinja2",
            Stage.AGENTS: "agents_prompt.jinja2",
        }
        tpl = self.env.get_template(template_map[stage])
        # Extended context for prompt_partials stage
        extra_context = {}
        if stage == Stage.PROMPT_PARTIALS:
            extra_context = {
                "extracted_roles_phases": self.results.get(Stage.ROLES_PHASES) or {},
                "extracted_state": self.results.get(Stage.STATE) or {},
                "existing_partials": (self.results.get(Stage.META_PARTIALS) or {}).get("prompt_partials", [])
            }
        header = f"You are parsing stage: {stage.value}."
        prompt = tpl.render(header=header, parsed_json=json.dumps(parsed, indent=2), extra_context=json.dumps(extra_context, indent=2))
        self.last_prompt = prompt
        return prompt

    async def _llm_call(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You output ONLY valid minified JSON. No explanations."},
            {"role": "user", "content": prompt}
        ]
        tracing_extra = {}
        return await self.llm.get_response(messages, tracing_extra)

    def run_current_stage(self, feedback: Optional[str] = None):
        with self.lock:
            stage = self.stages[self.current_index]
            if stage == Stage.FINALIZE:
                self._finalize()
                return
            self.state = StageState.WAITING
            prompt = feedback if feedback else self._render_stage_prompt(stage)
            thread = threading.Thread(target=self._stage_thread, args=(stage, prompt))
            thread.start()

    def _stage_thread(self, stage: Stage, prompt: str):
        import asyncio
        def ignore(loop, ctx):
            exc = ctx.get('exception')
            if isinstance(exc, RuntimeError) and str(exc) == 'Event loop is closed':
                return
            loop.default_exception_handler(ctx)
        loop = None
        try:
            loop = asyncio.new_event_loop()
            loop.set_exception_handler(ignore)
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self._llm_call(prompt))
            self.last_response = response
            self.state = StageState.PROCESSING
            self._process_response(stage, response)
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as e:
            self.errors[stage] = str(e)
            self.state = StageState.ERROR
        finally:
            if loop and not loop.is_closed():
                loop.close()

    # ------------- Response Processing / Validation -------------
    def _process_response(self, stage: Stage, response: str):
        try:
            data = json.loads(response)
        except Exception as e:
            self.errors[stage] = f"Invalid JSON: {e}. Raw: {response[:500]}"
            self.state = StageState.ERROR
            return
        ok, err = self._validate(stage, data)
        if not ok:
            self.errors[stage] = err
            self.state = StageState.ERROR
            return
        self.results[stage] = data
        self.errors[stage] = None
        self.state = StageState.SUCCESS

    def _validate(self, stage: Stage, data: Any) -> Tuple[bool, Optional[str]]:
        if stage == Stage.META_PARTIALS:
            if "meta" not in data or "prompt_partials" not in data:
                return False, "Missing meta or prompt_partials"
            if not isinstance(data["prompt_partials"], list):
                return False, "prompt_partials must be list"
            if not isinstance(data["meta"], dict):
                return False, "meta must be object"
            for p in data["prompt_partials"]:
                if not isinstance(p, dict) or "name" not in p or "content" not in p:
                    return False, "Each prompt_partial needs name & content"
        elif stage == Stage.ROLES_PHASES:
            required = ["roles", "phase_numbers", "actionable_phases", "role_task_map"]
            for r in required:
                if r not in data:
                    return False, f"Missing {r}"
        elif stage == Stage.STATE:
            if "state" not in data:
                return False, "Missing state"
            for sec in ["meta_information", "private_information", "public_information"]:
                if sec not in data["state"] or not isinstance(data["state"][sec], list):
                    return False, f"state.{sec} missing or not list"
        elif stage == Stage.PROMPT_PARTIALS:
            if "prompt_partials" not in data or not isinstance(data["prompt_partials"], list):
                return False, "Missing prompt_partials list"
            for pp in data["prompt_partials"]:
                if not isinstance(pp, dict) or "name" not in pp or "content" not in pp:
                    return False, "Each prompt_partial requires name & content"
        elif stage == Stage.ROLE_PROMPTS:
            if "role_prompts" not in data or not isinstance(data["role_prompts"], list):
                return False, "Missing role_prompts list"
            for rp in data["role_prompts"]:
                for k in ["role_identifier", "phase", "kind", "text"]:
                    if k not in rp:
                        return False, f"role_prompts entry missing {k}"
        elif stage == Stage.AGENTS:
            if "agents" not in data or not isinstance(data["agents"], list):
                return False, "Missing agents list"
            for a in data["agents"]:
                if "id" not in a or "role_ref" not in a:
                    return False, "agent missing id or role_ref"
        return True, None

    # ------------- Navigation -------------
    def next_stage(self):
        with self.lock:
            if self.current_index < len(self.stages) - 1:
                self.current_index += 1
                self.state = StageState.IDLE
                return self.stages[self.current_index]
            return None

    # ------------- Assembly -------------
    def _finalize(self):
        meta_block = self.results.get(Stage.META_PARTIALS, {})
        roles_phases = self.results.get(Stage.ROLES_PHASES, {})
        state_block = self.results.get(Stage.STATE, {})
        gen_partials_block = self.results.get(Stage.PROMPT_PARTIALS, {})
        role_prompts_block = self.results.get(Stage.ROLE_PROMPTS, {})
        agents_block = self.results.get(Stage.AGENTS, {})

        name = meta_block.get("meta", {}).get("name", "")
        if name == "cannot infer":
            name = ""
        description = meta_block.get("meta", {}).get("description", "")
        if description == "cannot infer":
            description = ""
        # Merge partials: meta stage first, then generated prompt_partials stage (overriding duplicates if generated has substantive content)
        merged_partials: Dict[str, str] = {}
        for source in [meta_block.get("prompt_partials", []), gen_partials_block.get("prompt_partials", [])]:
            for pp in source or []:
                n = pp.get("name")
                c = pp.get("content", "")
                if not n:
                    continue
                # Prefer non-empty / non 'cannot infer' content over existing
                existing = merged_partials.get(n)
                normalized_c = "" if c == "cannot infer" else c
                if existing is None or (not existing and normalized_c):
                    merged_partials[n] = normalized_c
        partials = [PromptPartial(name=k, content=v) for k, v in merged_partials.items()]

        # Roles & inference mapping (same as before)
        agent_roles: List[AgentRoleConfig] = []
        role_id_lookup: Dict[str, int] = {}
        for idx, r in enumerate(roles_phases.get("roles", []), start=1):
            raw_id = str(r.get("role_id_raw"))
            role_id_lookup[raw_id] = idx
            llm_type = r.get("llm_type")
            if llm_type == "cannot infer":
                llm_type = None
            llm_params = r.get("llm_params")
            if llm_params == "cannot infer" or not isinstance(llm_params, dict):
                llm_params = {}
            task_phases_inferred = r.get("task_phases_inferred")
            if task_phases_inferred == "cannot infer" or not isinstance(task_phases_inferred, list):
                task_phases_inferred = []
            agent_roles.append(AgentRoleConfig(
                role_id=idx,
                name=r.get("name", f"role_{idx}"),
                llm_type=llm_type,
                llm_params=llm_params,
                prompts=[],
                task_phases=task_phases_inferred,
                task_phases_excluded=[],
            ))

        # Prompts attach
        prompts_by_role: Dict[int, List[RolePromptEntry]] = {ar.role_id: [] for ar in agent_roles}
        for rp in role_prompts_block.get("role_prompts", []):
            rid_raw = str(rp.get("role_identifier"))
            text = rp.get("text", "")
            if text == "cannot infer":
                continue
            if rid_raw in role_id_lookup:
                kind = rp.get("kind")
                if kind in ("system", "user"):
                    prompts_by_role[role_id_lookup[rid_raw]].append(RolePromptEntry(key=kind, value=text))
        for ar in agent_roles:
            ar.prompts = prompts_by_role.get(ar.role_id, [])

        # State fields
        def build_fields(arr: List[Dict[str, Any]]) -> List[StateFieldConfig]:
            out: List[StateFieldConfig] = []
            for f in arr:
                nm = f.get("name") or f.get("id")
                if not nm:
                    continue
                tp = f.get("type", "")
                if tp == "cannot infer":
                    tp = ""
                dv = f.get("default")
                if dv == "cannot infer":
                    dv = None
                out.append(StateFieldConfig(name=nm, type=tp, default=dv))
            return out
        st = state_block.get("state", {})
        state_cfg = StateConfig(
            meta_information=build_fields(st.get("meta_information", [])),
            private_information=build_fields(st.get("private_information", [])),
            public_information=build_fields(st.get("public_information", [])),
        )

        # Agents
        agents_cfg: List[AgentMappingConfig] = []
        for a in agents_block.get("agents", []):
            role_ref = str(a.get("role_ref"))
            if role_ref in role_id_lookup:
                try:
                    agent_id_int = int(a.get("id"))
                except Exception:
                    continue
                agents_cfg.append(AgentMappingConfig(id=agent_id_int, role_id=role_id_lookup[role_ref]))

        self.experiment = ExperimentConfig(
            name=name,
            description=description,
            prompt_partials=partials,
            agent_roles=agent_roles,
            agents=agents_cfg,
            state=state_cfg,
            manager=ManagerConfig(),
            runner=RunnerConfig(),
        )
        self.results[Stage.FINALIZE] = self.experiment.to_template_context()
        self.state = StageState.SUCCESS

    def render_yaml(self, output_path: Optional[str] = None) -> str:
        if not self.experiment:
            raise RuntimeError("Finalize not executed")
        context = self.experiment.to_template_context()
        rendered = self.template.render(**context)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if output_path is None:
            base = os.path.splitext(os.path.basename(self.parsed_json_path))[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base}_final.yaml")
        with open(output_path, 'w') as f:
            f.write(rendered)
        return output_path

    def wait(self):
        while self.state == StageState.WAITING:
            time.sleep(0.25)

# ------------- CLI Runner -------------

def main():
    files = [os.path.join(PARSED_JSON_DIR, f) for f in os.listdir(PARSED_JSON_DIR) if f.endswith('.json')]
    if not files:
        print("No parsed JSON found.")
        return
    print("Available parsed specs:")
    for i, f in enumerate(files):
        print(f" [{i}] {f}")
    choice = 0
    selected = files[choice]
    print(f"Using: {selected}")
    interp = InterpretInStages(selected)

    while True:
        stage = interp.stages[interp.current_index]
        print(f"\n=== Stage: {stage.value} ===")
        if stage != Stage.FINALIZE:
            print("Prompt preview (truncated):")
            prompt_prev = interp._render_stage_prompt(stage)
            print(prompt_prev[:600] + ('...' if len(prompt_prev) > 600 else ''))
            interp.run_current_stage()
            interp.wait()
            if interp.state == StageState.ERROR:
                print(f"Error: {interp.errors[stage]}")
                feedback = "Ensure JSON strictly matches schema. Use 'cannot infer' where appropriate."
                interp.run_current_stage(feedback=feedback)
                interp.wait()
            if interp.state == StageState.SUCCESS:
                print("Result snippet:")
                print(json.dumps(interp.results[stage], indent=2)[:800])
            nxt = interp.next_stage()
            if not nxt:
                break
        else:
            interp.run_current_stage()
            if interp.state == StageState.SUCCESS:
                out_path = interp.render_yaml()
                print(f"YAML written to {out_path}")
            break

if __name__ == "__main__":
    main()
