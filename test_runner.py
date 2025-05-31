#!/usr/bin/env python3
"""
Test runner for the LLM Coordinator system.
This script provides an easy way to test the coordinator with different configurations.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional

# Assuming the coordinator is in the same directory
from llm_coordinator import LLMCoordinator, load_mock_responses

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_environment_xml(filename: str = "test_structure.xml"):
    """Create a simple test environment XML file."""
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<Structure>
    <Goal>
        <Block color="yellow" pos="(0, 0, 0)"/>
        <Block color="yellow" pos="(0, 1, 0)"/>
        <Block color="yellow" pos="(1, 0, 0)"/>
    </Goal>
    <Goal>
        <Block color="blue" pos="(0, 0, 1)"/>
        <Block color="blue" pos="(0, 0, 2)"/>
    </Goal>
    <Goal>
        <Block color="red" pos="(2, 0, 0)"/>
        <Block color="red" pos="(2, 1, 0)"/>
    </Goal>
</Structure>'''
    
    with open(filename, 'w') as f:
        f.write(xml_content)
    
    logger.info(f"Created test environment file: {filename}")
    return filename


def create_test_mock_responses(filename: str = "test_responses.json"):
    """Create test mock responses for different scenarios."""
    
    # Scenario 1: Basic cooperation
    basic_cooperation = {
        "agent1": [
            "I'll start building the yellow structure. place_block(block_type=yellow, pos=(0,0,0))",
            "Continuing with yellow blocks. place_block(block_type=yellow, pos=(0,1,0)) and send_chat(to=\"agent2\", message=\"I'm handling the yellow blocks, can you take the blue ones?\")",
            "place_block(block_type=yellow, pos=(1,0,0))",
            "My yellow structure is complete! wait() and send_chat(to=\"agent2\", message=\"Yellow blocks done! How's your progress?\")",
            "wait()",
            "I'll help agent3 if needed. wait()"
        ],
        "agent2": [
            "I'll work on the blue blocks. place_block(block_type=blue, pos=(0,0,1)) and send_chat(to=\"agent1\", message=\"Sure, I'll handle the blue blocks!\")",
            "Continuing with blue. place_block(block_type=blue, pos=(0,0,2))",
            "Blue structure complete! wait() and send_chat(to=\"agent3\", message=\"Blue blocks done. Need help with red?\")",
            "wait()",
            "wait()"
        ],
        "agent3": [
            "I see others are working. I'll focus on red blocks. place_block(block_type=red, pos=(2,0,0))",
            "Continuing with red. place_block(block_type=red, pos=(2,1,0)) and send_chat(to=\"agent2\", message=\"Thanks! Almost done with red.\")",
            "Red structure complete! wait()",
            "wait()"
        ]
    }
    
    # Scenario 2: With errors and recovery
    error_recovery = {
        "agent1": [
            "Let me place a yellow block. place_block(block_type=yellow, pos=(0,0,0))",
            "Oops, trying to place at wrong position. place_block(block_type=yellow, pos=(0,0,1))",
            "That failed, let me try the correct position. place_block(block_type=yellow, pos=(0,1,0))",
            "Good! Now the last one. place_block(block_type=yellow, pos=(1,0,0))",
            "wait()"
        ],
        "agent2": [
            "Working on blue blocks. place_block(block_type=blue, pos=(0,0,1))",
            "Next blue block. place_block(block_type=blue, pos=(0,0,2))",
            "wait()",
            "wait()"
        ]
    }
    
    # Scenario 3: Complex communication
    complex_communication = {
        "agent1": [
            "send_chat(to=\"agent2\", message=\"Let's coordinate our building. I'll do yellow, you do blue?\") and place_block(block_type=yellow, pos=(0,0,0))",
            "place_block(block_type=yellow, pos=(0,1,0)) and send_chat(to=\"agent3\", message=\"Agent3, can you handle the red blocks?\")",
            "place_block(block_type=yellow, pos=(1,0,0))",
            "send_chat(to=\"agent2\", message=\"Yellow complete!\") and send_chat(to=\"agent3\", message=\"How's the red structure?\") and wait()",
            "wait()"
        ],
        "agent2": [
            "send_chat(to=\"agent1\", message=\"Sounds good! Starting blue now.\") and place_block(block_type=blue, pos=(0,0,1))",
            "place_block(block_type=blue, pos=(0,0,2)) and send_chat(to=\"agent3\", message=\"We're making good progress!\")",
            "send_chat(to=\"agent1\", message=\"Blue done too!\") and wait()",
            "wait()"
        ],
        "agent3": [
            "send_chat(to=\"agent1\", message=\"Yes, I'll do red!\") and place_block(block_type=red, pos=(2,0,0))",
            "place_block(block_type=red, pos=(2,1,0)) and send_chat(to=\"agent2\", message=\"Indeed! Almost done here.\")",
            "send_chat(to=\"agent1\", message=\"Red structure complete!\") and wait()",
            "wait()"
        ]
    }
    
    scenarios = {
        "basic": basic_cooperation,
        "error": error_recovery,
        "communication": complex_communication
    }
    
    # Write all scenarios to file
    with open(filename, 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    logger.info(f"Created mock responses file: {filename}")
    return filename


def run_test(scenario: str = "basic", visualize: bool = False, max_turns: int = 20):
    """Run a test with the specified scenario."""
    
    # Create test files
    env_file = create_test_environment_xml()
    mock_file = create_test_mock_responses()
    
    # Load the specific scenario
    with open(mock_file, 'r') as f:
        all_scenarios = json.load(f)
    
    if scenario not in all_scenarios:
        logger.error(f"Unknown scenario: {scenario}. Available: {list(all_scenarios.keys())}")
        return False
    
    mock_responses = all_scenarios[scenario]
    
    # Determine number of agents from mock responses
    agent_names = list(mock_responses.keys())
    agent_configs = []
    
    for i, name in enumerate(agent_names):
        # Alternate between providers for variety
        provider = "openai" if i % 2 == 0 else "anthropic"
        model = "gpt-4" if provider == "openai" else "claude-3-opus-20240229"
        
        agent_configs.append({
            "name": name,
            "model": model,
            "provider": provider
        })
    
    logger.info(f"Running test scenario: {scenario}")
    logger.info(f"Agents: {agent_names}")
    
    # Create and run coordinator
    coordinator = LLMCoordinator(
        environment_xml=env_file,
        agent_configs=agent_configs,
        visualize=visualize,
        mock_responses=mock_responses
    )
    
    success = coordinator.run_game(max_turns=max_turns)
    
    # Clean up test files
    if os.path.exists(env_file):
        os.remove(env_file)
    if os.path.exists(mock_file):
        os.remove(mock_file)
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Test runner for LLM Coordinator')
    parser.add_argument('--scenario', default='basic', 
                       choices=['basic', 'error', 'communication'],
                       help='Test scenario to run')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable 3D visualization')
    parser.add_argument('--max-turns', type=int, default=20,
                       help='Maximum number of turns')
    parser.add_argument('--create-files-only', action='store_true',
                       help='Only create test files without running')
    
    args = parser.parse_args()
    
    if args.create_files_only:
        create_test_environment_xml()
        create_test_mock_responses()
        logger.info("Test files created. You can now run the coordinator manually.")
    else:
        success = run_test(
            scenario=args.scenario,
            visualize=args.visualize,
            max_turns=args.max_turns
        )
        
        if success:
            logger.info("Test completed successfully! Goal achieved.")
        else:
            logger.info("Test completed. Goal not achieved within turn limit.")


if __name__ == "__main__":
    main()