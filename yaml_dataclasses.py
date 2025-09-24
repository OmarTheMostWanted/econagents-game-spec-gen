from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal

"""Data model reflecting templates/econagents_template.yaml.jinja2
All fields marked optional in the template are modelled with Optional[...] or default values.
The top-level ExperimentConfig can be converted directly into a Jinja2 rendering context.
"""

# --- Prompt Partials ---
@dataclass
class PromptPartial:
    name: str
    content: str

# --- Event Handler ---
@dataclass
class EventHandler:
    event: str
    custom_code: Optional[str] = None
    custom_module: Optional[str] = None
    custom_function: Optional[str] = None

# --- Role Prompt Entry ---
@dataclass
class RolePromptEntry:
    key: str  # e.g. system, user, system_phase_2, user_phase_6 etc.
    value: str

# --- Agent Role Config ---
@dataclass
class AgentRoleConfig:
    role_id: int
    name: str
    llm_type: Optional[str] = None  # human filled (template default ChatOpenAI)
    llm_params: Dict[str, Any] = field(default_factory=dict)
    prompts: List[RolePromptEntry] = field(default_factory=list)
    task_phases: List[int] = field(default_factory=list)
    task_phases_excluded: List[int] = field(default_factory=list)

# --- Agent Mapping Config ---
@dataclass
class AgentMappingConfig:
    id: int
    role_id: int

# --- State Field Config ---
@dataclass
class StateFieldConfig:
    name: str
    type: str
    default: Any = None
    default_factory: Optional[str] = None
    event_key: Optional[str] = None
    exclude_from_mapping: bool = False
    optional: bool = False
    events: Optional[List[str]] = None
    exclude_events: Optional[List[str]] = None

# --- State Config ---
@dataclass
class StateConfig:
    meta_information: List[StateFieldConfig] = field(default_factory=list)
    private_information: List[StateFieldConfig] = field(default_factory=list)
    public_information: List[StateFieldConfig] = field(default_factory=list)

# --- Manager Config ---
@dataclass
class ManagerConfig:
    type: str = "TurnBasedPhaseManager"
    event_handlers: List[EventHandler] = field(default_factory=list)

# --- Runner Config ---
@dataclass
class RunnerConfig:
    type: str = "GameRunner"
    protocol: str = "ws"
    hostname: str = "localhost"
    path: str = "wss"
    port: int = 0
    game_id: int = 0
    logs_dir: str = "logs"
    log_level: str = "INFO"
    prompts_dir: str = "prompts"
    phase_transition_event: str = "phase-transition"
    phase_identifier_key: str = "phase"
    observability_provider: Optional[Literal["langsmith", "langfuse"]] = None
    continuous_phases: List[int] = field(default_factory=list)
    min_action_delay: int = 5
    max_action_delay: int = 10

# --- Experiment Config ---
@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    prompt_partials: List[PromptPartial] = field(default_factory=list)
    agent_roles: List[AgentRoleConfig] = field(default_factory=list)
    agents: List[AgentMappingConfig] = field(default_factory=list)
    state: StateConfig = field(default_factory=StateConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    def to_template_context(self) -> Dict[str, Any]:
        """Return a dict shaped for econagents_template.yaml.jinja2 rendering."""
        return {
            "experiment_name": self.name,
            "experiment_description": self.description,
            "prompt_partials": [asdict(p) for p in self.prompt_partials],
            "agent_roles": [
                {
                    "role_id": r.role_id,
                    "name": r.name,
                    "llm_type": r.llm_type,
                    "llm_params": r.llm_params,
                    "prompts": [asdict(pe) for pe in r.prompts],
                    "task_phases": r.task_phases,
                    "task_phases_excluded": r.task_phases_excluded,
                }
                for r in self.agent_roles
            ],
            "agents": [asdict(a) for a in self.agents],
            "state": {
                "meta_information": [asdict(f) for f in self.state.meta_information],
                "private_information": [asdict(f) for f in self.state.private_information],
                "public_information": [asdict(f) for f in self.state.public_information],
            },
            "manager": {
                "type": self.manager.type,
                "event_handlers": [asdict(eh) for eh in self.manager.event_handlers],
            },
            "runner": asdict(self.runner),
        }

# Utility factory helpers ---------------------------------------------------

def make_state_field_from_json(field_json: Dict[str, Any]) -> StateFieldConfig:
    """Create StateFieldConfig from a JSON field spec produced by parser.
    Supports keys: id/name, type, default, default_factory, event_key, exclude_from_mapping, optional, events, exclude_events.
    """
    return StateFieldConfig(
        name=field_json.get("name") or field_json.get("id"),
        type=field_json.get("type", ""),
        default=field_json.get("default"),
        default_factory=field_json.get("default_factory"),
        event_key=field_json.get("event_key"),
        exclude_from_mapping=bool(field_json.get("exclude_from_mapping", False)),
        optional=bool(field_json.get("optional", False)),
        events=field_json.get("events"),
        exclude_events=field_json.get("exclude_events"),
    )

__all__ = [
    "PromptPartial",
    "EventHandler",
    "RolePromptEntry",
    "AgentRoleConfig",
    "AgentMappingConfig",
    "StateFieldConfig",
    "StateConfig",
    "ManagerConfig",
    "RunnerConfig",
    "ExperimentConfig",
    "make_state_field_from_json",
]

