from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PhaseRoleTasks:
    """Tasks mapped to each role for a single phase."""
    phase: str
    phase_number: int
    actionable: bool
    role_tasks: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class PayoffConsequence:
    """Payoff consequence for a role's choice in a phase."""
    phase: str
    role: str
    choice: str
    payoff: str

@dataclass
class PhaseRoleMatrix:
    """Complete matrix: one entry per phase, plus payoff consequences."""
    phases: List[PhaseRoleTasks] = field(default_factory=list)
    payoff_consequences: List[PayoffConsequence] = field(default_factory=list)
