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
