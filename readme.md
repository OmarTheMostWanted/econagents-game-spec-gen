# A Structured Guide to Automatically Generating Valid EconAgents YAML Files from Natural Language Game Specifications

---

## Overview

The goal of this project is to Automatically generating valid EconAgents YAML files from natural language game specifications. With the aim to enable researchers and practitioners to convert human-readable descriptions of economic games into structured YAML configurations that are fully compatible with the EconAgents UI and its underlying Python framework.

**EconAgents** is an open-source Python library and UI ecosystem that facilitates connecting Large Language Model (LLM) agents to economic game servers for simulating experiments. Its architecture is modular, supporting the definition of agent roles, game state hierarchies, Jinja-based templated prompts, and flexible runner orchestration. The system leverages YAML configurations as the canonical format for game descriptions, agent behaviors, state management, and experiment orchestration.

Automating the translation from natural language game specs to EconAgents-compliant YAML introduces unique requirements:

- The ability to handle diverse and complex game structures (e.g., Prisoner’s Dilemma, Futarchy, Harberger Tax) and agent logic.
- Robustness in maintaining YAML syntax and semantics across hierarchical, nested, and conditional constructs.
- Security, to avoid injection vulnerabilities and guarantee safe, reproducible files.
- Human-in-the-loop controls for iterative refinement, error correction, and context retention.

Effectively, the translation pipeline must encapsulate natural language understanding, prompt engineering for structured output, schema-driven validation, and rigorous feedback loops—bridging the gap between game design and the strictures of machine-readable configuration.

---

## Translation Pipeline

### High-Level Pipeline Architecture

A translation pipeline for converting natural language economic game specifications into valid EconAgents YAML files consists of several modular stages.

1. **Input Preprocessing and Parsing:**  
   - Accept a plain natural language description of the game and preprocess for clarity, splitting into logical sections (e.g., Roles, Game Setup, Payoff Structure).
  
2. **Chunking and Semantic Segmentation:**  
   - Break down the specification into topic-constrained sections for more stable LLM output.
   - Create a phase-based role task matrix to map roles to specific tasks within each game phase.

3. **Prompt Construction and LLM Invocation:**  
   - Construct prompts (possibly chunk-wise) incorporating schema constraints and output format requirements.
   - Use a YAML template to guide the structure of the output.

4. **Schema-Driven YAML Generation:**  
   - Use LLM(s) to generate YAML for each segment, strongly steering the model toward schema-compliant output.
   - Optionally interleave human review at this point.

5. **Post-Processing and Merging:**  
   - Merge chunked outputs into a global YAML configuration, resolving anchors, references, and cross-chunk dependencies.

6. **Schema Validation:**  
   - Automatically validate the full YAML file with a strict schema validator (YAML Schema, JSON Schema-derived tools, etc.).
   - Validate empty task for roles.

7. **Feedback and Iterative Refinement:**  
   - If schema violations or semantic errors are detected, generate corrective prompts and repeat relevant steps with fine-tuned instructions.
   - Incorporate human feedback loops to refine prompts and outputs iteratively.

8. **Security and Finalization:**  
   - Sanitize inputs and outputs, apply safe YAML loading practices, check for injection or encoding attacks, and log for audit.

9. **Export/Delivery:**  
   - Output finalized YAML and use it to run a simulation
   - Feed the output logs back into the prompt-generation loop for further refinement.

---

# Standardizing Role-Phase-Task Structure for Economic Games in the EconAgents Framework

## Matrix Design: Conceptual Framework for Role-Phase-Task Standardization

### Why a Matrix Structure?

A **matrix** in the context of economic experiments is an abstract map (often represented as a table or a set of nested dictionaries) that clearly specifies, for each game:
- What *roles* exist (e.g., Trader, Observer, Policy-Maker),
- Which *phases* comprise a single round or the whole game (e.g., Offer Phase, Vote Phase, Settlement Phase),
- Which *tasks* are assigned (possibly zero, one, or multiple) per role in each phase.

Standardizing this structure is crucial for several reasons:
- Economists and experimentalists can define new games by populating a consistent schema, reducing ambiguity and setup time.
- LLM agents can be reliably instructed to act according to the matrix, interpreting their part in the game's logic and sequence.
- Downstream analysis, auditing, and reproducibility are enhanced by clarity and comparable data structures.

### Inspirations and Precedents

**1. Software and DSL Precedents:**  
The pipeline schema patterns seen in DevOps (Azure Pipelines: stages, jobs, tasks), project responsibility matrices (RACI, RASCI), and role-playing game design matrices offer compelling models for structured, phase-by-phase task allocation.

**2. oTree and Empirica Experiment Design:**  
These Python-based experiment platforms use explicit listings of roles, phases, and tasks in their configuration structure, allowing for clear separation, extensibility, and agent programmability.

**3. Prompt Engineering for LLMs:**  
Prompt templates, particularly those using Jinja2 or similar templating languages, enable structured, reusable, and highly manipulable definitions for LLM interactions, including explicit placeholders for role, phase, and task cues.

### The Matrix Pattern: A Schematic

A robust Role-Phase-Task matrix should:
- Be **typed and machine-validated**, allowing for schema checks and structural validation.
- Account for *optional*, *role-specific*, and *phase-specific* tasks.
- Allow typed or documented descriptions for roles, phases, and tasks (e.g., input format, output expectations, legal actions).

**Conceptual Example (Pseudo-YAML/Tabular View):**

| Role        | Phase        | Tasks             |
|-------------|--------------|-------------------|
| Trader      | Offer        | SubmitOffer       |
| Trader      | Vote         | CastVote          |
| Trader      | Settlement   | None              |
| Observer    | Offer        | ViewOffer         |
| Observer    | Vote         | None              |
| Observer    | Settlement   | RecordOutcome     |

This can be encoded as nested data for rigorous parsing, validation, and population—whether by LLM, code, or configuration file.

**YAML Sample (as in the Futarchy example):**

```yaml
roles:
  - trader
  - observer
phases:
  - offer
  - vote
  - settlement
tasks:
  trader:
    offer:
      - SubmitOffer
    vote:
      - CastVote
    settlement: []
  observer:
    offer:
      - ViewOffer
    vote: []
    settlement:
      - RecordOutcome
```

Such a matrix, in YAML or as a Python data class tree, provides all the information needed for both automated agent execution and human review.

### Enforcing Structure and Reliability

The matrix design must address:
- **Presence/Absence:** Appropriately mark roles/phases/tasks as optional or required, possibly using `null`, empty lists, or explicit "N/A" flags.
- **Order of Play:** Explicit ordering of phases and tasks for deterministic simulation steps.
- **Extensibility:** Ability to add fields for constraints, validation, or metadata (e.g., duration, dependencies, allowed values).
- **Validation:** Schema or code-based checks for completeness and logical consistency (e.g., every role must have an entry for every phase, even if “None”).
- **LLM Guidance:** Structured outputs and prompt compositions can leverage the matrix for deterministic or range-bounded LLM reasoning.

---

## Implementation Options: Comparing Approaches

Implementation in EconAgents must support:
- **Expressiveness:** Must accurately represent the complexity of multi-role, multi-phase, multi-task games.
- **Extensibility:** Support additions as new game designs, roles, or phase logic emerge.
- **Integration:** Alone or in combination, YAML schemas, code-based data structures, and prompt templates must dovetail with the EconAgents architecture.

We now examine each principal approach:

### 1. Code-Based (Data Classes / Strong Typing)

**Description:**  
Leverage Python data classes (`@dataclass`) to define roles, phases, and tasks, possibly using type annotations, enums, and validators. Each instance directly represents a game's matrix, enabling programmatic population, mutation, and validation.

**Strengths:**
- Built-in type safety, catch structure errors early.
- Enables IDE features (auto-complete, docstrings, refactors).
- Tight integration with validation libraries (e.g., pydantic for runtime validation).
- Facilitates schema generation for YAML or JSON export/import.
- Enables rich per-field documentation.

**Weaknesses:**
- Harder for non-programmers to read/write (unless auto-exported to human-centric text).
- Schema changes need code changes and redeployment.
- May require serializers/deserializers for YAML/JSON integration.

**Example:**
```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Task:
    name: str
    description: Optional[str] = None

@dataclass
class PhaseMatrix:
    phase: str
    tasks: Dict[str, List[Task]]  # Keyed by role name

@dataclass
class RolePhaseTaskMatrix:
    roles: List[str]
    phases: List[str]
    phase_matrices: List[PhaseMatrix]
```

**Use Cases:**  
Initial game design, programmatic generation, validation before runtime.

### 2. YAML (DSL/Config File-Based)

**Description:**  
Define the entire game matrix in a YAML file, following a strict schema. Human-readable and directly editable by experiment designers.

**Strengths:**
- Highly readable and modifiable by non-coders.
- Facilitates configuration as code (declarative approach).
- Supports versioning, diffing, and collaborative editing.
- Easily ingested for LLM prompt-generation or code-based loading.
- Aligns with existing EconAgents/Futarchy YAML patterns and documentation.

**Weaknesses:**
- Validation must be external (via code or YAML schema tools).
- Errors are often detected at load time, not edit time.
- Less flexible for computed/default values or complex logic unless templated.

**Example:**  
(See earlier in Matrix Design.)

### 3. Prompt-Based (Templated Prompts)

**Description:**  
Define prompts for LLMs that encode the matrix as part of their system or user messages, often leveraging templating languages like Jinja2 to fill in roles, phases, tasks at runtime. Templates can be static files or concatenated string components in code.

**Strengths:**
- Enables reuse and scaling (write once, compose many times).
- Can include detailed instructions to LLM (“You are [role], in phase [phase], your tasks are: [tasks] ...”).
- Supports conditional expansion, examples, and few-shot patterns.
- Encourages clear, structured, LLM-guided outputs for parsing and validation.

**Weaknesses:**
- Can drift from the "source of truth" if prompts and schemas diverge.
- Variable substitution errors can be difficult to debug.
- Over-complex templates become hard to maintain.

**Example:**

```jinja
You are a {{ role }}. 
Current phase: {{ phase }}.
{% if tasks %}
Your tasks for this phase:
{% for task in tasks %}
- {{ task.name }}: {{ task.description }}
{% endfor %}
{% else %}
You have no task during this phase.
{% endif %}
```

**Use Cases:**  
Generating agent-specific system/user prompts, rapid iteration, and batch simulation. Enforcing output structure (by instructing the LLM to use, e.g., JSON or bullet-list outputs matching the matrix).

### 4. Hybrid/Templated Code-Driven Pipelines

**Description:**  
Combine data classes/YAML matrices as the master definition, and drive prompt template expansion from this source (see *prompt managers* in LangChain or custom systems). This supports both formal validation and flexible downstream LLM usage.

**Strengths:**
- Maximum maintainability—single source of truth.
- Enables robust versioning, round-tripping, and audit trails.
- Bridges researcher/developer needs (non-coders define YAML, coders extend schemas and logic).
- Facilitates integration with prompt management and LLMOps tooling.

**Weaknesses:**
- Slightly higher complexity: must maintain mapping code.
- Requires discipline to keep code, YAML, and templates in sync.
- Onboarding can be steeper for new contributors.

### 5. Advanced Structured Output (LLM-Enforced Schemas)

**Description:**  
Couple prompts with output schemas (e.g., requiring the LLM to produce outputs matching a specified Pydantic schema or a strict JSON format). Recent advances allow enforcing output via LLM fine-tuning, constrained decoding, or tools like Outlines, Guidance, Guardrails, or vllm’s guided decoding.

**Strengths:**
- Guaranteed output conformance for downstream processing.
- Structured logging, chaining, and aggregation.
- Reduces errors—broken chains, hallucinated tasks, missing fields are programmatically handled.

**Weaknesses:**
- May require more engineering effort and up-to-date LLM models/tooling.
- May lead to rigidity in cases requiring open-ended/generated tasks.

---

### Comparative Summary Table

| Approach       | Human Accessibility | Rigor & Validation | LLM Guidance & Parsing | Flexibility | Optimized for EconAgents? |
|----------------|--------------------|---------------------|------------------------|-------------|--------------------------|
| Data Classes   | Medium             | High                | Medium                 | High        | Yes                      |
| YAML Schema    | High               | Medium (external)   | High                   | Medium      | Yes                      |
| Prompt Templ.  | High (with docs)   | Low-Medium          | High                   | High        | Yes                      |
| Hybrid (YAML + Code + Prompt) | High           | Highest               | Highest                | Highest    | Yes                      |
| Structured Output (e.g., Guidance/Outlines) | Medium/High | Highest        | Highest                | Medium     | Yes (cutting-edge)       |

---

## Recommended Approach

### Alignment with EconAgents Architectural Principles

The EconAgents framework is expressly designed to be **modular, extensible, and agent-centric**, using:
- AgentRole classes with phase and prompt customization (via naming conventions, methods, or explicit registration),
- YAML-driven configuration for easy game definition (compatible with non-coder workflows),
- Prompt template directories and resolution logic for fine-grained prompt customization per phase/role.

It also provides rich hooks for:
- Jinja2 prompt templates (for system/user prompts, resolved by naming convention),
- Explicit method override (for code-based customization per phase/task),
- Flexible role and phase eligibility (task_phases, excluded_phases),
- Output parsing, handler registration, and error fallback for full lifecycle management.

Given these architectural affordances, **the most robust and maintainable approach** is a *hybrid model* that:

1. **Defines the role-phase-task matrix as a YAML schema (or compatible JSON structure), adhering to a strict schema enforceable by tools and/or pydantic models.**
2. **Uses data classes and type checking to validate and load the matrix at simulation startup, offering programmatic guarantees.**
3. **Drives prompt template resolution and population using Jinja2 or similar, dynamically rendering phase/role/task instructions into executable prompts at runtime.**
4. **Feeds output expectations to the LLM, instructing it to respond in structured form (JSON or schema-bound), and validates/parses the LLM output using the expected types.**
5. **Supports chaining and correction: any errors, omissions, or malformed outputs are caught early, and LLMs are instructed (via re-prompting or chain-of-thought) to self-correct.**

### Actionable Implementation Steps

#### 1. Define a Matrix Schema

Create a schema (YAML or JSON) expressing the full (and possibly minimal) structure required:

```yaml
game_name: "Futarchy"
roles:
  - trader
  - observer
phases:
  - offer
  - vote
  - settlement
tasks:
  trader:
    offer:
      - name: SubmitOffer
        description: Submit a sell/buy offer.
    vote:
      - name: CastVote
        description: Vote on the outcome.
    settlement: []
  observer:
    offer:
      - name: ViewOffer
        description: Observe the offer phase.
    vote: []
    settlement:
      - name: RecordOutcome
        description: Log the final outcome.
meta:
  validation: strict
  version: 1.0
```
**Tip:** Ensure presence of all role-phase pairings, even when the task list is empty.

#### 2. Implement Data Classes (and/or Pydantic Models)

Define a set of data classes that mirror the schema and enforce type-checking (useful for IDEs, code-based extensions, and validation).

```python
from typing import Dict, List, Optional
from pydantic import BaseModel, validator

class TaskDef(BaseModel):
    name: str
    description: Optional[str]

class PhaseTasks(BaseModel):
    phase: str
    tasks: Dict[str, List[TaskDef]]

class GameMatrix(BaseModel):
    game_name: str
    roles: List[str]
    phases: List[str]
    tasks: Dict[str, Dict[str, List[TaskDef]]]
```
**Tip:** Write validators to ensure every phase exists for every role, as appropriate.

#### 3. Build Prompt Templates Using Jinja2

Populate a `prompts/` directory with:
- General system and user prompts (`role_system.jinja2`, `role_user.jinja2`)
- Phase/role-specific prompts (`role_system_phase_1.jinja2`, etc.)
- A fallback general prompt as required.

Each template must be able to expand variables for role, phase, and tasks.

**Example template:**

```jinja
System Prompt:
You are playing the role of {{ role }} in the game '{{ game_name }}'.
You are currently in the '{{ phase }}' phase.
Your task(s) for this phase:
{% if tasks|length > 0 %}
{% for task in tasks %}
- {{ task.name }}: {{ task.description }}
{% endfor %}
{% else %}
No active tasks this phase.
{% endif %}
Respond accurately and only with your decision or action, following the output format specified below.

{# Optionally, include output format instruction for JSON #}
Output Format:
{{ output_schema }}
```

#### 4. Parse and Validate Outputs

- Always instruct LLM agents to reply matching the expected output schema (instruct them with explicit, repeatable format instructions).
- Use Pydantic models, Outlines, LangChain, or similar to parse and enforce structure; employ Guardrails or Guidance for runtime validation where strictness is essential.

#### 5. Integrate Matrix, Templates, and Code in EconAgents

- Load the YAML (or JSON) matrix at runtime.
- Instantiate agent roles with phase eligibility according to the matrix.
- During each phase, dynamically resolve the prompt template for that phase/role pair, injecting task and context from the matrix.
- Route output through a validator, retry/chain with LLM if schema is not satisfied, and log all interactions for reproducibility and debugging.

**Tip:** Document every matrix version, template, and handler for auditability and experimental repeatability.

### Implementation Example in EconAgents

**Role Initialization (Python):**
```python
class TraderRole(AgentRole):
    task_phases = ["offer", "vote"]
    def __init__(self, ...):
        ...
    def get_offer_system_prompt(self, state):
        # Load the matrix/tasks for offer phase from YAML
        tasks = self.get_matrix_tasks('trader', 'offer')
        return render_template('trader_system_offer.jinja2', {"tasks": tasks, "phase": "offer", ...})
```

**Phase Handling:**
- On each phase, call the resolved system/user prompt from the template, filled with tasks from the matrix.
- If the agent has no tasks, prompt accordingly.

**YAML Integration:**
- Define all game structures as separate config files, versioned in the codebase.

**Prompt Rendering:**
- Use Jinja2 or similar both for prompt expansion and (optionally) for configuration generation (e.g., create YAML game definitions from spreadsheet forms or again via prompt templates).

#### 6. Performance, Maintainability, and Extensibility Considerations

- **Maintainability:** Single-source YAML + code models → easy to update as games evolve.
- **Extensibility:** New games require only filling out the matrix and adding any needed templates.
- **LLM compatibility:** Prompts are always aligned with simulation structure, minimizing hallucination and confusion.
- **Performance:** Pre-render templates or prompt sections to reduce LLM calls; cache and reuse as needed.
- **Versioning/Auditing:** All changes to matrices or templates versioned; logs can be associated with exact schema and prompt bundle for later analysis/reproduction.

---

## Integration with EconAgents: Best Practices and Alignment

### How This Integrates with EconAgents

The recommended approach aligns directly with the architecture and philosophy of EconAgents, as evidenced by:

- **AgentRole class support for phase-based prompt/template resolution:**  
  The system is built to automatically seek the relevant system/user prompt per phase, by handler or template filename, matching the matrix’s role-phase-task mapping.

- **YAML as a first-class schema carrier:**  
  Games and experiments are shipped in YAML, with easy editing for researchers and seamless integration with code loaders/validators.

- **Templates and code modularization:**  
  Prompts are stored in files, registered, and resolved at runtime, allowing for composable, understandable, and maintainable simulation logic.

- **LangChain or Guidance-Style Structured Output:**  
  When higher assurance is desired, LLM outputs can be parsed/validated against model schemas, enabling error recovery, retries, and chaining, as in advanced prompt pipeline setups.

### Standardization Guidance

To cement standardized practices:
- **Establish a canonical YAML schema for all new games, including explicit version, meta/documentation fields, and example sections.**
- **Mandate per-game `prompts/` directories with required base and phase-specific templates.**
- **Provide codified data classes and/or Pydantic schemas for loading and runtime validation.**
- **Document and test every new game with a "dry run": check matrix completeness, prompt rendering, and LLM output parsing.**
- **Implement and version prompt templates alongside their matrix definitions, employing automation and linters for QA.**
- **Create user/project guides, including best practices for adding new phases, roles, or tasks, and for writing and testing prompt templates.**

---

## Summary of Benefits and Final Recommendations

**By implementing a robust Role-Phase-Task Matrix using the hybrid YAML+DataClass+Templated Prompt approach, EconAgents will achieve:**
- Maximum clarity for both developers and experimentalists designing new games.
- Reproducible, auditable, and scalable experiment configurations.
- Consistency and reliability in LLM-guided simulation, with error-minimizing structured prompt interfaces.
- Rapid onboarding for new use cases or agent/game designs.
- Alignment with both code-based extensibility and prompt-driven LLM workflows.

**Action Items:**
- Develop and publish the canonical matrix schema and corresponding Python data classes.
- Refactor existing games to use the standardized matrix where not already present.
- Create an automated validation + prompt generation toolchain as part of the EconAgents codebase.
- Train users (economists, experiment designers) on writing matrix-compatible YAML and authoring corresponding templates using best practices in prompt design and schema validation.

By adhering to these principles and practices, the EconAgents project positions itself as a gold standard for economic simulation research—blending computational rigor, human accessibility, and LLM-native extensibility in the age of agent-based economic experimentation.