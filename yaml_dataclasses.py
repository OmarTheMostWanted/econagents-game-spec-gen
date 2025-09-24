from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal, Callable

# --- Event Handler ---

@dataclass
class EventHandler:
    event: str
    custom_code: Optional[str] = None
    custom_module: Optional[str] = None
    custom_function: Optional[str] = None

# --- Agent Role Config ---

@dataclass
class AgentRoleConfig:
    role_id: int
    name: str
    llm_type: str = "ChatOpenAI"
    llm_params: Dict[str, Any] = field(default_factory=dict)
    prompts: List[Dict[str, str]] = field(default_factory=list)
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
    hostname: str = ""
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
    prompt_partials: List[Dict[str, str]] = field(default_factory=list)
    agent_roles: List[AgentRoleConfig] = field(default_factory=list)
    agents: List[AgentMappingConfig] = field(default_factory=list)
    state: StateConfig = field(default_factory=StateConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
