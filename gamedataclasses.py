from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class RolePhaseTasks:
    """Tasks mapped to each phase for a single role."""
    role: str
    phase_tasks: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class RolePhaseMatrix:
    """Complete matrix: one entry per role."""
    roles: List[RolePhaseTasks] = field(default_factory=list)
