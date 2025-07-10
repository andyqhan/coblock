# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating LLM collaboration in a Minecraft-like block-building environment. The project implements a multi-agent system where different language models work together to construct 3D structures from XML blueprints.

## Key Components

### Core Architecture
- **CoblockEnvironment** (`coblock_environment.py`): The main game engine managing 3D block physics, gravity constraints, and state validation
- **LLMCoordinator** (`llm_coordinator.py`): Orchestrates multiple AI agents, handles turn-based gameplay, and manages agent communication
- **ModelComparisonOrchestrator** (`model_comparison_orchestrator.py`): Runs systematic comparisons between different LLM models across various tasks

### Agent System
- Agents take turns placing/removing blocks and can communicate via chat
- Each agent has private goals and inventories
- Supports multiple LLM providers: OpenAI, Anthropic, Google Gemini
- Implements gravity constraints (blocks must be supported by ground or adjacent blocks)

### Structure Format
- Goals are defined in XML format with block colors and 3D positions
- Goals are distributed round-robin among agents
- Example structures include bridges, houses, and simple L-shapes

## Common Commands

### Running Single Games
```bash
# Run a game with two agents on a structure
python llm_coordinator.py structures/bridge.xml --max-turns 40 --visualize

# Run with mock responses for testing
python llm_coordinator.py structures/simple_l.xml --mock test_responses.json
```

### Model Comparisons
```bash
# Run systematic model comparisons
python model_comparison_orchestrator.py --structures structures/bridge.xml structures/house.xml --trials 3 --max-turns 40

# Generate test structures
python model_comparison_orchestrator.py --create-test-structures
```

### Environment Testing
```bash
# Test environment with interactive CLI
python coblock_environment.py structures/bridge.xml --visualize

# Commands in CLI: place red (0,0,0) agent1 | remove (0,0,0) agent1 | state | goal | check
```

### Analysis
```bash
# Analyze comparison results
python analyze_comparison_results.py comparison_results/results.json

# Run specific test scenarios. This is the main entrypoint.
python run_comparison.py --structures structures/bridge.xml --trials 5 --max-turns 30 --output-dir comparison_results_opus4_bridge_20250707
```

## Development Notes

DO NOT run Python programs without first consulting the user. Since they send AI API requests, they can be expensive.

### Environment Variables Required
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Anthropic models  
- `GEMINI_API_KEY`: For Google Gemini models

### Key Constraints
- Blocks must follow gravity rules (supported by ground or adjacent blocks)
- Agents can only place blocks from their inventory
- Agents can only remove blocks they placed
- Game ends when all goals completed or agents vote to end

### Test Structure Format
The XML structures define collaborative building tasks:
- `bridge.xml`: Two towers connected by a horizontal bridge (requires coordination)
- `house.xml`: Multi-story house with foundation, walls, and roof (complex coordination)
- `simple_l.xml`: Basic L-shape (simple coordination test)

### Agent Communication
- Agents communicate via `send_chat(to="agent_name", message="text")` 
- Messages are delivered on the recipient's next turn
- No shared knowledge of goals or inventories between agents

### Visualization
- 3D matplotlib visualization shows goal blocks as wireframes and current blocks as solid
- Real-time updates during gameplay
- Can be enabled with `--visualize` flag
