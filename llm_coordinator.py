import os
import ast
import re
import json
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from dotenv import load_dotenv
import time
from abc import ABC, abstractmethod

# Import the environment
from coblock_environment import CoblockEnvironment

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Represents an LLM agent with its state."""
    name: str
    model: str
    provider: str  # 'openai', 'anthropic', etc.
    inventory: Dict[str, int] = field(default_factory=dict)
    goals: List[ET.Element] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    chat_messages: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    failed_actions: List[Dict[str, Any]] = field(default_factory=list)
    voted_end_game: bool = False
    
    def add_to_inventory(self, color: str, count: int = 1):
        """Add blocks to agent's inventory."""
        self.inventory[color] = self.inventory.get(color, 0) + count
    
    def remove_from_inventory(self, color: str, count: int = 1) -> bool:
        """Remove blocks from agent's inventory. Returns True if successful."""
        if self.inventory.get(color, 0) >= count:
            self.inventory[color] -= count
            if self.inventory[color] == 0:
                del self.inventory[color]
            return True
        return False
    
    def get_inventory_xml(self) -> str:
        """Get inventory in XML format."""
        if not self.inventory:
            return "<Inventory>\n</Inventory>"
        
        lines = ["<Inventory>"]
        for color, count in self.inventory.items():
            lines.append(f'    <Block color="{color}" count={count}/>')
        lines.append("</Inventory>")
        return "\n".join(lines)
    
    def get_goals_xml(self) -> str:
        """Get goals in XML format."""
        if not self.goals:
            return "<Goals>\n</Goals>"
        
        lines = ["<Goals>"]
        for i, goal in enumerate(self.goals):
            lines.append(f'    <Goal id="{i+1}">')
            for block in goal.findall('Block'):
                color = block.get('color')
                pos = block.get('pos')
                lines.append(f'        <Block color="{color}" pos="{pos}"/>')
            lines.append('    </Goal>')
        lines.append("</Goals>")
        return "\n".join(lines)
    
    
    def get_dialogues_xml(self, other_agents: List[str]) -> str:
        """Get dialogues with other agents in XML format."""
        if not self.chat_messages:
            return ""
        
        dialogues = []
        for other_agent in other_agents:
            if other_agent in self.chat_messages and self.chat_messages[other_agent]:
                lines = ["<Dialogue>"]
                for msg in self.chat_messages[other_agent]:
                    sender = msg['sender']
                    to = msg['to']
                    message = msg['message']
                    lines.append(f'    <Message sender="{sender}" to="{to}" message="{message}"/>')
                lines.append("</Dialogue>")
                dialogues.append("\n".join(lines))
        
        return "\n".join(dialogues) if dialogues else ""


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def send_messages(self, messages: List[Dict[str, str]], model: str) -> str:
        """Send messages to LLM and get response."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def send_messages(self, messages: List[Dict[str, str]], model: str) -> str:
        """Send messages to OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""
    
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def send_messages(self, messages: List[Dict[str, str]], model: str) -> str:
        """Send messages to Anthropic API."""
        try:
            # Convert OpenAI format to Anthropic format
            system_message = None
            converted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    converted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.7,
                system=system_message if system_message else "",
                messages=converted_messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self):
        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def send_messages(self, messages: List[Dict[str, str]], model: str) -> str:
        """Send messages to Gemini API."""
        try:
            # Convert OpenAI format to Gemini format
            system_instruction = None
            conversation_history = []

            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                else:
                    # Gemini uses "user" and "model" roles (instead of "assistant")
                    role = "model" if msg["role"] == "assistant" else msg["role"]
                    conversation_history.append({
                        "role": role,
                        "parts": [{"text": msg["content"]}]
                    })

            # Configure the model with system instruction if provided
            config = self.types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000,
                system_instruction=system_instruction if system_instruction else None,
                thinking_config=self.types.ThinkingConfig(include_thoughts=True),
            )

            # For single turn conversations, use generate_content
            if len(conversation_history) == 1:
                response = self.client.models.generate_content(
                    model=model,
                    contents=conversation_history[0]["parts"][0]["text"],
                    config=config
                )
                return_string = ""
                for part in response.candidates[0].content.parts:  # type: ignore
                    if not part.text:
                        continue
                    if part.thought:  # a bool flag
                        return_string += f"[Thought] {part.text}\n"
                    else:
                        return_string += f"[Answer] {part.text}\n"
                return return_string

            # For multi-turn conversations, use chat
            else:
                # Separate the last message from history
                *history, last_message = conversation_history
                # if len(conversation_history) == 5:
                #     logger.info(f"System prompt: {system_instruction}")
                #     logger.info(f"Conversation history at 5: {conversation_history}")

                # Create chat session with history
                chat = self.client.chats.create(
                    model=model,
                    config=config,
                    history=history if history else []
                )

                # Send the last message
                response = chat.send_message(last_message["parts"][0]["text"])
                return_string = ""
                for part in response.candidates[0].content.parts:  # type: ignore
                    if not part.text:
                        continue
                    if part.thought:  # a bool flag
                        return_string += f"[Thought] {part.text}\n"
                    else:
                        return_string += f"[Answer] {part.text}\n"
                return return_string

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, mock_responses: Dict[str, List[str]]):
        self.mock_responses = mock_responses
        self.response_indices = defaultdict(int)
        
    def send_messages(self, messages: List[Dict[str, str]], model: str) -> str:
        """Return mocked responses."""
        # Extract agent name from the messages
        agent_name = None
        for msg in messages:
            if "You are" in msg.get("content", ""):
                match = re.search(r"You are (\w+)", msg["content"])
                if match:
                    agent_name = match.group(1)
                    break
        
        if agent_name and agent_name in self.mock_responses:
            responses = self.mock_responses[agent_name]
            idx = self.response_indices[agent_name]
            if idx < len(responses):
                response = responses[idx]
                self.response_indices[agent_name] += 1
                return response
        
        return "wait()"


@dataclass
class WorldAction:
    """Represents an action taken in the world."""
    agent: str
    action_type: str  # 'place_block', 'remove_block', 'wait'
    details: Dict[str, Any]
    turn: int
    success: bool


class LLMCoordinator:
    """Coordinates LLM agents in the block environment."""
    
    def __init__(self, environment_xml: str, agent_configs: List[Dict[str, str]], 
                 visualize: bool = False, mock_responses: Optional[Dict[str, List[str]]] = None,
                 perfect_information: bool = False, show_diff: bool = False):
        """
        Initialize the coordinator.
        
        Args:
            environment_xml: Path to the XML file describing the structure
            agent_configs: List of dicts with 'name', 'model', and 'provider' for each agent
            visualize: Whether to enable 3D visualization
            mock_responses: Optional mock responses for testing
            perfect_information: If True, agents get full goal information instead of just their own goals
            show_diff: If True, show difference between current and target structure in turn prompts
        """
        # Initialize environment
        self.env = CoblockEnvironment(environment_xml, visualize=visualize)
        
        # Parse goals from XML
        self.goals = self._parse_goals(environment_xml)
        self.perfect_information = perfect_information
        self.show_diff = show_diff
        
        # Initialize agents
        self.agents: Dict[str, Agent] = {}
        self.agent_order: List[str] = []
        self._init_agents(agent_configs)
        
        # Assign goals to agents
        self._assign_goals()
        
        # Initialize providers
        self.providers: Dict[str, LLMProvider] = {}
        self._init_providers(mock_responses)
        
        # Game state
        self.current_agent_idx = 0
        self.turn = 0
        self.world_actions: List[WorldAction] = []
        self.pending_messages: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.consecutive_end_votes = 0
        
        # Statistics
        self.total_actions = 0
        self.failed_actions_count = 0
        
        logger.info(f"LLMCoordinator initialized with {len(self.agents)} agents")
    
    def _parse_goals(self, xml_file: str) -> List[ET.Element]:
        """Parse goals from the XML file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return root.findall('Goal')
    
    def _init_agents(self, agent_configs: List[Dict[str, str]]):
        """Initialize agents from configurations."""
        for config in agent_configs:
            name = config['name']
            model = config['model']
            provider = config['provider']
            blocks: Dict[str, int] = config.get('blocks', {})  # type: ignore
            
            # Initialize agent with starting inventory
            agent = Agent(name=name, model=model, provider=provider)
            
            # Give each agent some starting blocks
            if blocks:
                for color, count in blocks.items():
                    agent.add_to_inventory(color, count)
            else:
                for color in ['red', 'blue', 'yellow', 'green', 'brown', 'gray']:
                    agent.add_to_inventory(color, 10)
            
            self.agents[name] = agent
            self.agent_order.append(name)
            
            logger.info(f"Initialized agent {name} with model {model} from {provider}")
    
    def _assign_goals(self):
        """Assign goals to agents in round-robin fashion."""
        for i, goal in enumerate(self.goals):
            agent_idx = i % len(self.agent_order)
            agent_name = self.agent_order[agent_idx]
            self.agents[agent_name].goals.append(goal)
            logger.info(f"Assigned goal {i+1} to {agent_name}")
    
    def _init_providers(self, mock_responses: Optional[Dict[str, List[str]]] = None):
        """Initialize LLM providers."""
        if mock_responses:
            self.providers['mock'] = MockProvider(mock_responses)
            # Override all providers to use mock
            for agent in self.agents.values():
                agent.provider = 'mock'
        else:
            # Initialize real providers
            if any(a.provider == 'openai' for a in self.agents.values()):
                self.providers['openai'] = OpenAIProvider()
            
            if any(a.provider == 'anthropic' for a in self.agents.values()):
                self.providers['anthropic'] = AnthropicProvider()

            if any(a.provider == 'google' for a in self.agents.values()):
                self.providers['google'] = GeminiProvider()
    
    def get_current_agent(self) -> Agent:
        """Get the current agent whose turn it is."""
        return self.agents[self.agent_order[self.current_agent_idx]]
    
    def get_next_agent_name(self) -> str:
        """Get the name of the next agent in the circular list."""
        next_idx = (self.current_agent_idx + 1) % len(self.agent_order)
        return self.agent_order[next_idx]
    
    def get_full_goals_xml(self) -> str:
        """Get all goals in XML format (for perfect information mode)."""
        if not self.goals:
            return "<Goals>\n</Goals>"
        
        lines = ["<Goals>"]
        for i, goal in enumerate(self.goals):
            lines.append(f'    <Goal id="{i+1}">')
            for block in goal.findall('Block'):
                color = block.get('color')
                pos = block.get('pos')
                lines.append(f'        <Block color="{color}" pos="{pos}"/>')
            lines.append('    </Goal>')
        lines.append("</Goals>")
        return "\n".join(lines)
    
    def _construct_initial_prompt(self, agent: Agent) -> str:
        """Construct the initial task introduction prompt."""
        prompt = """# Task Summary
You will act as a Minecraft player collaborating with another agent to build a structure with a blueprint. You need to use the following commands to interact with the Minecraft world:
## Place a red block at the position of (x: 1, y: 0, z: 1).
place_block(block_type=red, pos=(1,0,1))
## Chat with a partner
send_chat(to="agent2", message="Hello, partner")
## Destroy the block at the position of (3,0,3). You will receive the block back in your inventory.
remove_block(pos=(3,0,3))
## Wait for your turn
wait()
## Vote to end the game (use when you think you're done with the game)
end_game()

# World state format
At each turn, you will receive information about the world state in this format:
<World>
    <Block color="yellow" pos="(0,0,0)" owner="agent1"/>
    <Block color="yellow" pos="(0,0,1)" owner="agent2"/>
</World>

# Inventory format
At each turn, you will receive information about your inventory in this format:
<Inventory>
    <Block color="red" count=3/>
    <Block color="yellow" count=1/>
</Inventory>

# Your individual task
"""

# # Message format
# At each turn, you will get your message history with every other agent, if you have one, in this format (in this example, assume you are "agent1"):
# <Dialogue>
#     <Message sender="agent1" to="agent2" message="Hello!"/>
#     <Message sender="agent2" to="agent1" message="Nice to meet you!"/>
# </Dialogue>
# <Dialogue>
#     <Message sender="agent1" to="agent3" message="My name is agent1, how about you?"/>
#     <Message sender="agent3" to="agent1" message="I'm agent3!"/>
# </Dialogue>

        # Add goal information based on perfect_information setting
        if self.perfect_information:
            prompt += "\n\n# Full structure information (all goals)\n"
            prompt += self.get_full_goals_xml()
            prompt += "# Your individual task\n"
            prompt += agent.get_goals_xml()
            goal_instructions = """\n\nKeep in mind the following rules:
- You can only place blocks from *your* inventory.
- You must always send an action command (`place_block`, `remove_block`, or `wait`) on every turn. You may optionally send a `send_chat` command, which is the *only* way to communicate with your partner.
- You and your partner can see ALL goals above. You are responsible for your individual task, but you should coordinate to build the complete structure efficiently."""
        else:
            prompt += "# Your individual task\n"
            prompt += agent.get_goals_xml()
            goal_instructions = """\n\nKeep in mind the following rules:
- You can only place blocks from *your* inventory.
- You must always send an action command (`place_block`, `remove_block`, or `wait`) on every turn. You may optionally send a `send_chat` command, which is the *only* way to communicate with your partner.
- Your partner does not know what your goal is, nor do they know what blocks you have. You do not know what your partner's goal is. You have different goals than your partner."""
        
        diff_instructions = ""
        if self.show_diff:
            if self.perfect_information:
                diff_instructions = """
- You will also be provided a comparison between the goal structure and current structure in the world. This comparison helps you to accurately track the progress and decide the next block to build.
<ComparisonResult>
<MissingBlocks>
# any missing blocks from the full structure
</MissingBlocks>
<ExtraBlocks>
# any extra blocks placed that shouldn't be there
</ExtraBlocks>
</ComparisonResult>
The blocks in MissingBlocks should be built. The blocks in ExtraBlocks should be removed.
"""
            else:
                diff_instructions = """
- You will also be provided a comparison between your goal structure and the blocks you have placed. This comparison helps you to accurately track the progress and decide the next block to build.
<ComparisonResult>
<MissingBlocks>
# any missing blocks from your individual goals
</MissingBlocks>
<ExtraBlocks>
# any extra blocks you've placed that shouldn't be there
</ExtraBlocks>
</ComparisonResult>
The blocks in MissingBlocks should be built. The blocks in ExtraBlocks should be removed.
"""

        additional_instructions = goal_instructions + diff_instructions + """
- Block placements *must* adhere to gravity: every block you place has to be connected either to the ground (y=0) or another block (which is eventually connected to the ground). *Blocks do not have to be directly supported*: they can be supported by an adjacent block. For example, if there are blocks at (0,0,0) and (0,1,0), you can place a block at (1,1,0), even though there's no block at (1,0,0), because the (1,1,0) block is supported by the (0,1,0) block.
- Block placements will *fail* if they do not adhere to gravity or another block is already in that location.
- The y-axis is the vertical axis. y is the second number in the three-tuple positions. If y=0, then it's ground level (and will adhere to gravity), otherwise, you need a supporting block. For example, (1,0,1) is x=1, y=0, z=1, and is placeable without any other blocks. (0,2,1) is x=0, y=2, z=1, and is not placeable without supporting blocks.
- You may only destroy blocks that you have placed. If you need to destroy a block someone else placed, ask them with `send_chat`.
- Success is when you and your partner have placed all and *only* the blocks for both of your goals (with no extra blocks). The game will end automatically when your goals are complete.
- Use `end_game` when you believe all goals have been completed or they are impossible. If all agents vote to end the game consecutively, the game will end.
"""
        prompt += additional_instructions
        prompt += f"\n\nYou are {agent.name}."
        return prompt
    
    def _get_world_state_xml(self) -> str:
        """Get the current world state in XML format."""
        state = self.env.get_current_state()
        if not state:
            return "<World>\n</World>"
        
        lines = ["<World>"]
        for pos, info in sorted(state.items()):
            color = info['color']
            owner = info['owner']
            lines.append(f'    <Block color="{color}" pos="{pos}" owner="{owner}"/>')
        lines.append("</World>")
        return "\n".join(lines)
    
    def _get_diff_xml(self, agent: Agent) -> str:
        """Get the diff between current state and goals in XML format."""
        lines = ["<ComparisonResult>"]
        
        if self.perfect_information:
            # Show diff for the complete structure
            missing_blocks = self.env.get_missing_blocks()
            extra_blocks = self.env.get_extra_blocks()
        else:
            # Show diff only for this agent's goals

            agent_goals = {}
            for goal in agent.goals:
                for block in goal.findall('Block'):
                    color = block.get('color')
                    pos = ast.literal_eval(block.get('pos', ''))
                    agent_goals[pos] = color
            missing_blocks = self.env.get_agent_missing_blocks(agent.name, agent_goals)
            extra_blocks = self.env.get_agent_extra_blocks(agent.name, agent_goals)
        
        # Add missing blocks
        lines.append("<MissingBlocks>")
        if missing_blocks:
            for pos, color in sorted(missing_blocks.items()):
                lines.append(f'    <Block color="{color}" pos="{pos}"/>')
        lines.append("</MissingBlocks>")
        
        # Add extra blocks
        lines.append("<ExtraBlocks>")
        if extra_blocks:
            for pos, info in sorted(extra_blocks.items()):
                color = info['color']
                owner = info['owner']
                lines.append(f'    <Block color="{color}" pos="{pos}" owner="{owner}"/>')
        lines.append("</ExtraBlocks>")
        
        lines.append("</ComparisonResult>")
        return "\n".join(lines)
    
    def _get_recent_actions_text(self, agent_name: str) -> str:
        """Get text describing recent actions by other agents."""
        if not self.world_actions:
            return ""
        
        # Find the last turn this agent played
        last_turn = -1
        for action in reversed(self.world_actions):
            if action.agent == agent_name:
                last_turn = action.turn
                break
        
        # Get actions since then by other agents
        recent_actions = [a for a in self.world_actions 
                         if a.turn > last_turn and a.agent != agent_name]
        
        if not recent_actions:
            return ""
        
        lines = ["Recent actions by other agents:"]
        for action in recent_actions:
            if action.action_type == 'place_block':
                lines.append(f"- {action.agent} placed a {action.details['color']} block at {action.details['pos']}")
            elif action.action_type == 'remove_block':
                lines.append(f"- {action.agent} removed a block at {action.details['pos']}")
            elif action.action_type == 'wait':
                lines.append(f"- {action.agent} waited")
        
        return "\n".join(lines)
    
    def _construct_turn_prompt(self, agent: Agent) -> str:
        """Construct the prompt for an agent's turn."""
        parts = []
        
        # Add current state information
        parts.append("=== Current Turn ===\n")
        parts.append("Your inventory:")
        parts.append(agent.get_inventory_xml())
        parts.append("\nCurrent world state:")
        parts.append(self._get_world_state_xml())
        
        # Add diff information if enabled
        if self.show_diff:
            parts.append("\nComparison result:")
            parts.append(self._get_diff_xml(agent))
        
        # Add any pending messages
        if agent.name in self.pending_messages and self.pending_messages[agent.name]:
            parts.append("\nNew messages:")
            for msg in self.pending_messages[agent.name]:
                parts.append(f"From {msg['sender']}: {msg['message']}")
            # Clear pending messages after showing them
            self.pending_messages[agent.name] = []
        
        # Add dialogue history
        # other_agents = [a for a in self.agent_order if a != agent.name]
        # dialogues = agent.get_dialogues_xml(other_agents)
        # if dialogues:
        #     parts.append("\nMessage history:")
        #     parts.append(dialogues)

        # # Add recent actions
        # recent_actions = self._get_recent_actions_text(agent.name)
        # if recent_actions:
        #     parts.append("\n" + recent_actions)

        # Add any failed actions from last turn
        if agent.failed_actions:
            parts.append("\nYour previous action failed:")
            for failed in agent.failed_actions:
                parts.append(f"- {failed['reason']}")
            agent.failed_actions = []  # Clear after showing
        
        parts.append("\nWhat is your next action? (Remember to use one of: place_block, remove_block, wait, end_game, and optionally send_chat)")
        
        return "\n".join(parts)
    
    def _parse_llm_response(self, response: str, agent: Agent) -> List[Dict[str, Any]]:
        """Parse commands from LLM response."""
        commands = []
        
        # Parse place_block command
        place_match = re.search(r'place_block\s*\(\s*block_type\s*=\s*\"?(\w+)\"?\s*,\s*pos\s*=\s*\(\"?([^)]+)\"?\)\s*\)', response)
        if place_match:
            color = place_match.group(1)
            pos_str = place_match.group(2)
            pos = tuple(map(int, pos_str.split(',')))
            commands.append({
                'type': 'place_block',
                'color': color,
                'pos': pos
            })
        
        # Parse remove_block command
        remove_match = re.search(r'remove_block\s*\(\s*pos\s*=\s*\(([^)]+)\)\s*\)', response)
        if remove_match:
            pos_str = remove_match.group(1)
            pos = tuple(map(int, pos_str.split(',')))
            commands.append({
                'type': 'remove_block',
                'pos': pos
            })
        
        # Parse wait command
        if re.search(r'wait\s*\(\s*\)', response):
            commands.append({'type': 'wait'})

        # Parse end_game command
        if re.search(r'end_game\s*\(\s*\)', response):
            commands.append({'type': 'end_game'})
        
        # Parse send_chat commands (can have multiple)
        chat_matches = re.finditer(r'send_chat\s*\(\s*to\s*=\s*"([^"]+)"\s*,\s*message\s*=\s*"([^"]+)"\s*\)', response)
        for match in chat_matches:
            to_agent = match.group(1)
            message = match.group(2)
            commands.append({
                'type': 'send_chat',
                'to': to_agent,
                'message': message
            })
        
        return commands
    
    def _execute_command(self, command: Dict[str, Any], agent: Agent) -> bool:
        """Execute a parsed command. Returns True if successful."""
        cmd_type = command['type']
        
        if cmd_type == 'place_block':
            color = command['color']
            pos = command['pos']
            
            # Check inventory
            if color not in agent.inventory or agent.inventory[color] <= 0:
                reason = f"You don't have any {color} blocks in your inventory"
                agent.failed_actions.append({'command': command, 'reason': reason})
                logger.warning(f"{agent.name} tried to place {color} block but has none")
                return False
            
            # Try to place block
            success, reason = self.env.place_block(color, pos, agent.name)
            if success:
                agent.remove_from_inventory(color)
                self.world_actions.append(WorldAction(
                    agent=agent.name,
                    action_type='place_block',
                    details={'color': color, 'pos': pos},
                    turn=self.turn,
                    success=True
                ))
                logger.info(f"{agent.name} successfully placed {color} block at {pos}")
            else:
                if not reason:
                    reason = f"Cannot place block at {pos} (position occupied or violates gravity)"
                agent.failed_actions.append({'command': command, 'reason': reason})
                self.failed_actions_count += 1
                logger.warning(f"{agent.name} failed to place block at {pos}")
            
            self.consecutive_end_votes = 0
            return success
        
        elif cmd_type == 'remove_block':
            pos = command['pos']
            state = self.env.get_current_state()
            if pos in state:
                color = state[pos]['color']
            success = self.env.remove_block(pos, agent.name)
            
            if success:
                # Add block back to inventory (assuming we know the color)
                if color:
                    agent.add_to_inventory(color)
                
                self.world_actions.append(WorldAction(
                    agent=agent.name,
                    action_type='remove_block',
                    details={'pos': pos},
                    turn=self.turn,
                    success=True
                ))
                logger.info(f"{agent.name} successfully removed block at {pos}")
            else:
                reason = f"Cannot remove block at {pos} (no block there, not your block, or would break gravity)"
                agent.failed_actions.append({'command': command, 'reason': reason})
                self.failed_actions_count += 1
                logger.warning(f"{agent.name} failed to remove block at {pos}")
            
            self.consecutive_end_votes = 0
            return success
        
        elif cmd_type == 'wait':
            self.world_actions.append(WorldAction(
                agent=agent.name,
                action_type='wait',
                details={},
                turn=self.turn,
                success=True
            ))
            logger.info(f"{agent.name} waited")
            self.consecutive_end_votes = 0
            return True

        elif cmd_type == 'end_game':
            self.consecutive_end_votes += 1
            self.world_actions.append(WorldAction(
                agent=agent.name,
                action_type='end_game',
                details={},
                turn=self.turn,
                success=True
            ))
            return True

        elif cmd_type == 'send_chat':
            to_agent = command['to']
            message = command['message']
            
            if to_agent in self.agents:
                # Add to pending messages for the recipient
                msg_data = {
                    'sender': agent.name,
                    'to': to_agent,
                    'message': message
                }
                self.pending_messages[to_agent].append(msg_data)
                
                # Add to both agents' chat history
                agent.chat_messages[to_agent].append(msg_data)
                self.agents[to_agent].chat_messages[agent.name].append(msg_data)
                
                logger.info(f"{agent.name} sent message to {to_agent}: {message}")
            else:
                logger.warning(f"{agent.name} tried to send message to unknown agent {to_agent}")
            
            return True

        return False
    
    def play_turn(self):
        """Execute one turn of the game."""
        time.sleep(1)  # hopefully this makes Anthropic's rate limiter happy
        agent = self.get_current_agent()
        logger.info(f"\n=== Turn {self.turn + 1}: {agent.name}'s turn ===")
        
        # Construct messages for the LLM
        messages = agent.messages.copy()
        
        # Add initial prompt if first turn
        if not messages:
            messages.append({
                "role": "system",
                "content": self._construct_initial_prompt(agent)
            })
        
        # Add current turn prompt
        turn_prompt = self._construct_turn_prompt(agent)
        messages.append({
            "role": "user",
            "content": turn_prompt
        })
        
        # Get LLM response
        provider = self.providers.get(agent.provider) or self.providers.get('mock')
        response = provider.send_messages(messages, agent.model)
        
        logger.debug(f"{agent.name} messages: {messages}")
        logger.info(f"{agent.name} response: {response}")
        
        # Update agent's message history
        agent.messages = messages
        agent.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Parse and execute commands
        commands = self._parse_llm_response(response, agent)
        
        # Separate world action commands from chat commands
        world_commands = [c for c in commands if c['type'] in ['place_block', 'remove_block', 'wait', 'end_game']]
        chat_commands = [c for c in commands if c['type'] == 'send_chat']
        
        # Execute chat commands (always allowed)
        for cmd in chat_commands:
            self._execute_command(cmd, agent)
        
        # Execute world action (only one allowed)
        if len(world_commands) > 1:
            logger.warning(f"{agent.name} tried to execute multiple world actions, only first will be executed")
        
        if world_commands:
            self._execute_command(world_commands[0], agent)
            self.total_actions += 1
        else:
            # If no world action, treat as wait
            self._execute_command({'type': 'wait'}, agent)
            self.total_actions += 1
        
        # Move to next agent
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agent_order)
        self.turn += 1

        # Check if goal is achieved
        if self.env.is_goal_achieved():
            logger.info("=== GOAL ACHIEVED! ===")
            return True

        # Check if all agents voted to end game
        if self.consecutive_end_votes >= len(self.agent_order):
            logger.info("=== ALL AGENTS VOTED TO END GAME ===")
            return True

        return False
    
    def run_game(self, max_turns: int = 100):
        """Run the game until completion or max turns."""
        logger.info("Starting game...")
        
        for _ in range(max_turns):
            game_over = self.play_turn()  # play_turn returns True if goal achieved or agents voted to end.
            if game_over:
                if not self.env.is_goal_achieved() and self.consecutive_end_votes >= len(self.agent_order):
                    # Game failed if agents voted to end without the goal achieved
                    return False
                self._print_statistics()
                return True

            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        logger.info(f"Game ended after {max_turns} turns without achieving goal")
        self._print_statistics()
        return False
    
    def _print_statistics(self):
        """Print game statistics."""
        logger.info("\n=== Game Statistics ===")
        logger.info(f"Total turns: {self.turn}")
        logger.info(f"Total actions: {self.total_actions}")
        logger.info(f"Failed actions: {self.failed_actions_count}")
        logger.info(f"Success rate: {((self.total_actions - self.failed_actions_count) / self.total_actions * 100):.1f}%")
        
        for agent_name, agent in self.agents.items():
            successful_actions = len([a for a in self.world_actions 
                                    if a.agent == agent_name and a.success])
            logger.info(f"{agent_name}: {successful_actions} successful actions")


def load_mock_responses(json_file: str) -> Dict[str, List[str]]:
    """Load mock responses from a JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LLM Coordinator')
    parser.add_argument('environment_xml', help='Path to environment XML file')
    parser.add_argument('--mock', help='Path to mock responses JSON file')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--max-turns', type=int, default=20, help='Maximum number of turns')
    args = parser.parse_args()
    
    # Example agent configurations
    agent_configs = [
        # {"name": "agent1", "model": "gpt-4", "provider": "openai"},
        {"name": "agent1", "model": "claude-3-5-haiku-20241022", "provider": "anthropic",
         "blocks": {
             "yellow": 0,
             "blue": 2,
         }},
        {"name": "agent2", "model": "claude-3-5-haiku-20241022", "provider": "anthropic",
         "blocks": {
             "yellow": 2,
             "blue": 0,
         }}
    ]
    
    # Load mock responses if provided
    mock_responses = None
    if args.mock:
        mock_responses = load_mock_responses(args.mock)
    
    # Create and run coordinator
    coordinator = LLMCoordinator(
        environment_xml=args.environment_xml,
        agent_configs=agent_configs,
        visualize=args.visualize,
        mock_responses=mock_responses
    )
    
    coordinator.run_game(max_turns=args.max_turns)
