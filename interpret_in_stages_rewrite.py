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

# Directory constants
PROMPTS_DIR = "prompts/interpret2"
TEMPLATE_DIR = "templates"
TEMPLATE_FILE = "econagents_template.yaml.jinja2"
PARSED_JSON_DIR = "output/parse_out"
OUTPUT_DIR = "output/experiment_yaml"

# ---------------- Stages -----------------
class Stage(Enum):
    META = "meta"
    AGENT_ROLES = "agent_roles"
    STATE = "state"
    ROLE_PROMPTS = "role_prompts"
    AGENTS = "agents"
    MANAGER = "manager"
    RUNNER = "runner"
    FINALIZE = "finalize"

class StageState(Enum):
    IDLE = auto()
    WAITING = auto()
    PROCESSING = auto()
    SUCCESS = auto()
    ERROR = auto()

# ---------------- Interpreter --------------
class FreshInterpreter:
    """New interpreter built from scratch per latest requirements.
    Each YAML template section has a dedicated stage & prompt file (except prompt_partials which are copied directly).
    Optional sections are still requested; LLM can respond with sentinel 'cannot infer'.
    """
    def __init__(self, parsed_json_path: str):
        load_dotenv()
        self.parsed_json_path = parsed_json_path
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(api_key=self.api_key)
        self.env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
        self.template_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)
        self.template = self.template_env.get_template(TEMPLATE_FILE)

        self.stages: List[Stage] = [
            Stage.META,
            Stage.AGENT_ROLES,
            Stage.STATE,
            Stage.ROLE_PROMPTS,
            Stage.AGENTS,
            Stage.MANAGER,
            Stage.RUNNER,
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

    # ---------- Helpers ----------
    def _load_parsed(self) -> Dict[str, Any]:
        with open(self.parsed_json_path, 'r') as f:
            return json.load(f)

    def _prompt_filename(self, stage: Stage) -> Optional[str]:
        mapping = {
            Stage.META: "meta_prompt.jinja2",
            Stage.AGENT_ROLES: "roles_prompt.jinja2",
            Stage.STATE: "state_prompt.jinja2",
            Stage.ROLE_PROMPTS: "role_prompts_prompt.jinja2",
            Stage.AGENTS: "agents_prompt.jinja2",
            Stage.MANAGER: "manager_prompt.jinja2",
            Stage.RUNNER: "runner_prompt.jinja2",
        }
        return mapping.get(stage)

    def _render_prompt(self, stage: Stage) -> str:
        parsed = self._load_parsed()
        fname = self._prompt_filename(stage)
        if not fname:
            return "No prompt for this stage (FINALIZE)."
        tpl = self.env.get_template(fname)
        header = f"You are parsing stage: {stage.value}."
        prompt = tpl.render(header=header, parsed_json=json.dumps(parsed, indent=2))
        self.last_prompt = prompt
        return prompt

    async def _llm_call(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Return ONLY valid JSON. No explanations."},
            {"role": "user", "content": prompt}
        ]
        tracing_extra = {}
        return await self.llm.get_response(messages, tracing_extra)

    def run_stage(self, feedback: Optional[str] = None):
        with self.lock:
            stage = self.stages[self.current_index]
            if stage == Stage.FINALIZE:
                self._finalize()
                return
            self.state = StageState.WAITING
            prompt = feedback if feedback else self._render_prompt(stage)
            thread = threading.Thread(target=self._stage_thread, args=(stage, prompt))
            thread.start()

    def _stage_thread(self, stage: Stage, prompt: str):
        import asyncio
        def ignore_event_loop_closed(loop, ctx):
            exc = ctx.get('exception')
            if isinstance(exc, RuntimeError) and str(exc) == 'Event loop is closed':
                return
            loop.default_exception_handler(ctx)
        loop = None
        try:
            loop = asyncio.new_event_loop()
            loop.set_exception_handler(ignore_event_loop_closed)
            asyncio.set_event_loop(loop)
            resp = loop.run_until_complete(self._llm_call(prompt))
            self.last_response = resp
            self.state = StageState.PROCESSING
            self._process_response(stage, resp)
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as e:
            self.errors[stage] = str(e)
            self.state = StageState.ERROR
        finally:
            if loop and not loop.is_closed():
                loop.close()

    # ---------- Processing & Validation ----------
    def _process_response(self, stage: Stage, response: str):
        try:
            data = json.loads(response)
        except Exception as e:
            self.errors[stage] = f"Invalid JSON: {e}. Raw: {response[:300]}"
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
        if stage == Stage.META:
            if 'meta' not in data or not isinstance(data['meta'], dict):
                return False, "Missing meta object"
            for k in ['name', 'description']:
                if k not in data['meta']:
                    return False, f"meta missing {k}"
        elif stage == Stage.AGENT_ROLES:
            if 'agent_roles' not in data or not isinstance(data['agent_roles'], list):
                return False, "Missing agent_roles list"
            for idx, r in enumerate(data['agent_roles']):
                for k in ['raw_role_id', 'name', 'llm_type', 'llm_params', 'task_phases', 'task_phases_excluded', 'notes']:
                    if k not in r:
                        return False, f"agent_roles[{idx}] missing {k}"
            if 'phase_number_map' not in data or 'actionable_phase_numbers' not in data:
                return False, "Missing phase_number_map or actionable_phase_numbers"
        elif stage == Stage.STATE:
            if 'state' not in data:
                return False, "Missing state"
            for sec in ['meta_information', 'private_information', 'public_information']:
                if sec not in data['state'] or not isinstance(data['state'][sec], list):
                    return False, f"state.{sec} missing or not list"
        elif stage == Stage.ROLE_PROMPTS:
            if 'role_prompts' not in data or not isinstance(data['role_prompts'], list):
                return False, "Missing role_prompts list"
            for i, rp in enumerate(data['role_prompts']):
                for k in ['raw_role_id', 'phase_name', 'kind', 'content']:
                    if k not in rp:
                        return False, f"role_prompts[{i}] missing {k}"
        elif stage == Stage.AGENTS:
            if 'agents' not in data or not isinstance(data['agents'], list):
                return False, "Missing agents list"
            for i, a in enumerate(data['agents']):
                for k in ['id', 'raw_role_ref']:
                    if k not in a:
                        return False, f"agents[{i}] missing {k}"
        elif stage == Stage.MANAGER:
            if 'manager' not in data or not isinstance(data['manager'], dict):
                return False, "Missing manager object"
            if 'type' not in data['manager'] or 'event_handlers' not in data['manager']:
                return False, "manager missing type or event_handlers"
            if not isinstance(data['manager']['event_handlers'], list):
                return False, "manager.event_handlers not list"
        elif stage == Stage.RUNNER:
            if 'runner' not in data or not isinstance(data['runner'], dict):
                return False, "Missing runner object"
            required = ["type","protocol","hostname","path","port","game_id","logs_dir","log_level","prompts_dir","phase_transition_event","phase_identifier_key","observability_provider","continuous_phases","min_action_delay","max_action_delay"]
            for k in required:
                if k not in data['runner']:
                    return False, f"runner missing {k}"
        return True, None

    # ---------- Navigation ----------
    def next_stage(self) -> Optional[Stage]:
        if self.current_index < len(self.stages) - 1:
            self.current_index += 1
            self.state = StageState.IDLE
            return self.stages[self.current_index]
        return None

    # ---------- Final Assembly ----------
    def _finalize(self):
        parsed = self._load_parsed()
        # Copy prompt_partials directly from parsed JSON (no LLM) if present
        raw_partials = parsed.get('prompt_partials', []) or []
        prompt_partials: List[PromptPartial] = []
        for pp in raw_partials:
            if isinstance(pp, dict) and 'name' in pp and 'content' in pp:
                prompt_partials.append(PromptPartial(name=pp['name'], content=pp['content']))

        meta = self.results.get(Stage.META, {}).get('meta', {}) if self.results.get(Stage.META) else {}
        roles_block = self.results.get(Stage.AGENT_ROLES, {})
        state_block = self.results.get(Stage.STATE, {})
        role_prompts_block = self.results.get(Stage.ROLE_PROMPTS, {})
        agents_block = self.results.get(Stage.AGENTS, {})
        manager_block = self.results.get(Stage.MANAGER, {})
        runner_block = self.results.get(Stage.RUNNER, {})

        # Build roles
        agent_roles: List[AgentRoleConfig] = []
        raw_roles = roles_block.get('agent_roles', []) if roles_block else []
        role_id_lookup: Dict[str,int] = {}
        for idx, r in enumerate(raw_roles, start=1):
            raw_id = str(r.get('raw_role_id'))
            role_id_lookup[raw_id] = idx
            llm_type = r.get('llm_type')
            if llm_type == 'cannot infer':
                llm_type = None
            llm_params = r.get('llm_params') if isinstance(r.get('llm_params'), dict) else {}
            task_phases = r.get('task_phases') if isinstance(r.get('task_phases'), list) else []
            agent_roles.append(AgentRoleConfig(
                role_id=idx,
                name=r.get('name','cannot infer'),
                llm_type=llm_type,
                llm_params=llm_params,
                prompts=[],
                task_phases=task_phases,
                task_phases_excluded=[],
            ))
        # Attach role prompts
        prompts_by_role: Dict[int,List[RolePromptEntry]] = {ar.role_id: [] for ar in agent_roles}
        for rp in role_prompts_block.get('role_prompts', []) if role_prompts_block else []:
            rid = str(rp.get('raw_role_id'))
            if rid in role_id_lookup:
                content = rp.get('content','')
                if content:
                    prompts_by_role[role_id_lookup[rid]].append(RolePromptEntry(key=rp.get('kind'), value=content))
        for ar in agent_roles:
            ar.prompts = prompts_by_role.get(ar.role_id, [])

        # State
        def convert_field(f: Dict[str,Any]) -> Optional[StateFieldConfig]:
            name = f.get('name') or f.get('id')
            if not name:
                return None
            return StateFieldConfig(name=name, type=f.get('type',''), default=f.get('default'))
        st_json = state_block.get('state', {}) if state_block else {}
        state_cfg = StateConfig(
            meta_information=[cf for f in st_json.get('meta_information', []) if (cf:=convert_field(f))],
            private_information=[cf for f in st_json.get('private_information', []) if (cf:=convert_field(f))],
            public_information=[cf for f in st_json.get('public_information', []) if (cf:=convert_field(f))],
        )

        # Agents mapping
        agents_cfg: List[AgentMappingConfig] = []
        for a in agents_block.get('agents', []) if agents_block else []:
            rid = str(a.get('raw_role_ref'))
            if rid in role_id_lookup:
                try:
                    aid = int(a.get('id'))
                except Exception:
                    continue
                agents_cfg.append(AgentMappingConfig(id=aid, role_id=role_id_lookup[rid]))

        # Manager & Runner (keep 'cannot infer' strings intact)
        manager_cfg = ManagerConfig(type=manager_block.get('manager', {}).get('type','cannot infer') if manager_block else 'cannot infer')
        runner_json = runner_block.get('runner', {}) if runner_block else {}
        runner_cfg = RunnerConfig(
            type=runner_json.get('type','cannot infer'),
            protocol=runner_json.get('protocol','cannot infer'),
            hostname=runner_json.get('hostname','cannot infer'),
            path=runner_json.get('path','cannot infer'),
            port=runner_json.get('port',0) if isinstance(runner_json.get('port'), int) else 0,
            game_id=runner_json.get('game_id',0) if isinstance(runner_json.get('game_id'), int) else 0,
            logs_dir=runner_json.get('logs_dir','cannot infer'),
            log_level=runner_json.get('log_level','cannot infer'),
            prompts_dir=runner_json.get('prompts_dir','cannot infer'),
            phase_transition_event=runner_json.get('phase_transition_event','cannot infer'),
            phase_identifier_key=runner_json.get('phase_identifier_key','cannot infer'),
            observability_provider=None if runner_json.get('observability_provider') in (None,'cannot infer') else runner_json.get('observability_provider'),
            continuous_phases=runner_json.get('continuous_phases',[]) if isinstance(runner_json.get('continuous_phases'), list) else [],
            min_action_delay=runner_json.get('min_action_delay',5) if isinstance(runner_json.get('min_action_delay'), int) else 5,
            max_action_delay=runner_json.get('max_action_delay',10) if isinstance(runner_json.get('max_action_delay'), int) else 10,
        )

        self.experiment = ExperimentConfig(
            name=meta.get('name','cannot infer'),
            description=meta.get('description','cannot infer'),
            prompt_partials=prompt_partials,
            agent_roles=agent_roles,
            agents=agents_cfg,
            state=state_cfg,
            manager=manager_cfg,
            runner=runner_cfg,
        )
        self.results[Stage.FINALIZE] = self.experiment.to_template_context()
        self.state = StageState.SUCCESS

    def write_yaml(self) -> str:
        if not self.experiment:
            raise RuntimeError("Finalize before writing YAML")
        ctx = self.experiment.to_template_context()
        rendered = self.template.render(**ctx)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.parsed_json_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}_fresh.yaml")
        with open(out_path, 'w') as f:
            f.write(rendered)
        return out_path

    def wait(self):
        while self.state == StageState.WAITING:
            time.sleep(0.25)

    def retry_with_feedback(self, feedback: str):
        stage = self.stages[self.current_index]
        base = self.last_prompt or ""
        prior = self.last_response or ""
        prompt = f"Previous prompt:\n{base}\n\nPrevious response:\n{prior}\n\nFeedback:\n{feedback}\n\nRe-answer strictly as valid JSON."
        self.run_stage(feedback=prompt)

# ------------- CLI Runner -------------

def main():
    files = [os.path.join(PARSED_JSON_DIR, f) for f in os.listdir(PARSED_JSON_DIR) if f.endswith('.json')]
    if not files:
        print("No parsed JSON files found.")
        return
    print("Parsed JSON specs:")
    for i,f in enumerate(files):
        print(f" [{i}] {f}")
    choice = 0
    selected = files[choice]
    print(f"Using {selected}")
    interp = FreshInterpreter(selected)

    while True:
        stage = interp.stages[interp.current_index]
        print(f"\n=== Stage: {stage.value} ===")
        if stage != Stage.FINALIZE:
            preview = interp._render_prompt(stage)
            print("Prompt preview (truncated):")
            print(preview[:600] + ('...' if len(preview)>600 else ''))
            interp.run_stage()
            interp.wait()
            if interp.state == StageState.ERROR:
                print("Error:", interp.errors[stage])
                interp.retry_with_feedback("Correct JSON schema. Use 'cannot infer' where appropriate.")
                interp.wait()
            if interp.state == StageState.SUCCESS:
                print("Result snippet:")
                print(json.dumps(interp.results[stage], indent=2)[:800])
            nxt = interp.next_stage()
            if not nxt:
                break
        else:
            interp.run_stage()
            if interp.state == StageState.SUCCESS:
                out_path = interp.write_yaml()
                print(f"YAML written to {out_path}")
            break

if __name__ == "__main__":
    main()

