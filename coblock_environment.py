import xml.etree.ElementTree as ET
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Block:
    """Represents a block in the Minecraft-like environment."""
    color: str
    pos: Tuple[int, int, int]
    owner: str = ""
    
    # Pointers to adjacent blocks (6 faces)
    neighbors: Dict[str, Optional['Block']] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = {
                'up': None,      # +z
                'down': None,    # -z
                'north': None,   # +y
                'south': None,   # -y
                'east': None,    # +x
                'west': None     # -x
            }
    
    def __hash__(self):
        return hash(self.pos)
    
    def __eq__(self, other):
        if isinstance(other, Block):
            return self.pos == other.pos
        return False


class BlockVisualizer:
    """3D visualization of the block environment."""
    
    def __init__(self, environment: 'CoblockEnvironment'):
        self.env = environment
        self.fig = None
        self.ax = None
        self.is_closed = False
        
        # Color mapping
        self.color_map = {
            'red': '#FF0000',
            'blue': '#0000FF',
            'yellow': '#FFFF00',
            'green': '#00FF00',
            'brown': '#8B4513',
            'gray': '#808080',
            'stone': '#A9A9A9',
            'wood': '#DEB887',
            'sandstone': '#F4A460',
            'gold': '#FFD700',
            'white': '#FFFFFF',
            'black': '#000000',
            'orange': '#FFA500',
            'purple': '#800080'
        }
        
        # Initialize the plot
        self._init_plot()
    
    def _init_plot(self):
        """Initialize the 3D plot."""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Handle window close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        # Initial draw
        self.update()
    
    def _on_close(self, event):
        """Handle window close event."""
        self.is_closed = True
    
    def _get_color(self, color_name: str, alpha: float = 1.0) -> tuple:
        """Convert color name to RGBA tuple."""
        hex_color = self.color_map.get(color_name.lower(), '#CCCCCC')
        rgb = mcolors.hex2color(hex_color)
        return (*rgb, alpha)
    
    def _create_cube_vertices(self, pos: Tuple[int, int, int], size: float = 0.9):
        """Create vertices for a cube at the given position."""
        x, y, z = pos
        half_size = size / 2
        
        # Define vertices of a cube centered at (x, y, z)
        vertices = [
            [x - half_size, y - half_size, z - half_size],
            [x + half_size, y - half_size, z - half_size],
            [x + half_size, y + half_size, z - half_size],
            [x - half_size, y + half_size, z - half_size],
            [x - half_size, y - half_size, z + half_size],
            [x + half_size, y - half_size, z + half_size],
            [x + half_size, y + half_size, z + half_size],
            [x - half_size, y + half_size, z + half_size]
        ]
        
        # Define the 6 faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
        ]
        
        return faces
    
    def _draw_block(self, block: Block, alpha: float = 1.0, size: float = 0.9):
        """Draw a single block."""
        faces = self._create_cube_vertices(block.pos, size)
        color = self._get_color(block.color, alpha)
        
        # Create polygon collection
        poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, 
                               edgecolor='black', linewidth=0.5)
        self.ax.add_collection3d(poly)
    
    def _draw_wireframe_block(self, block: Block):
        """Draw a wireframe block for the goal state."""
        x, y, z = block.pos
        size = 0.95
        half_size = size / 2
        
        # Define edges of the wireframe cube
        edges = [
            [(x - half_size, y - half_size, z - half_size), (x + half_size, y - half_size, z - half_size)],
            [(x + half_size, y - half_size, z - half_size), (x + half_size, y + half_size, z - half_size)],
            [(x + half_size, y + half_size, z - half_size), (x - half_size, y + half_size, z - half_size)],
            [(x - half_size, y + half_size, z - half_size), (x - half_size, y - half_size, z - half_size)],
            [(x - half_size, y - half_size, z + half_size), (x + half_size, y - half_size, z + half_size)],
            [(x + half_size, y - half_size, z + half_size), (x + half_size, y + half_size, z + half_size)],
            [(x + half_size, y + half_size, z + half_size), (x - half_size, y + half_size, z + half_size)],
            [(x - half_size, y + half_size, z + half_size), (x - half_size, y - half_size, z + half_size)],
            [(x - half_size, y - half_size, z - half_size), (x - half_size, y - half_size, z + half_size)],
            [(x + half_size, y - half_size, z - half_size), (x + half_size, y - half_size, z + half_size)],
            [(x + half_size, y + half_size, z - half_size), (x + half_size, y + half_size, z + half_size)],
            [(x - half_size, y + half_size, z - half_size), (x - half_size, y + half_size, z + half_size)]
        ]
        
        color = self._get_color(block.color, 0.3)
        for edge in edges:
            self.ax.plot3D(*zip(*edge), color=color[:3], alpha=0.5, linewidth=2)
    
    def update(self):
        """Update the visualization with current state."""
        if self.is_closed:
            return
        
        # Clear the plot
        self.ax.clear()
        
        # Set labels again after clearing
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Block Environment Visualization\n(Wireframe: Goal, Solid: Current)')
        
        # Draw goal blocks as wireframes
        for block in self.env.goal_graph.values():
            self._draw_wireframe_block(block)
        
        # Draw current blocks as solid
        for block in self.env.current_graph.values():
            self._draw_block(block)
        
        # Calculate bounds for all blocks
        all_positions = list(self.env.goal_graph.keys()) + list(self.env.current_graph.keys())
        if all_positions:
            x_coords = [p[0] for p in all_positions]
            y_coords = [p[1] for p in all_positions]
            z_coords = [p[2] for p in all_positions]
            
            # Set axis limits with some padding
            padding = 1
            self.ax.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
            self.ax.set_ylim(min(y_coords) - padding, max(y_coords) + padding)
            self.ax.set_zlim(-0.5, max(z_coords) + padding)
        else:
            # Default view if no blocks
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_zlim(-0.5, 3)
        
        # Draw grid at z=0
        xx, yy = np.meshgrid(range(-5, 6), range(-5, 6))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')
        
        # Update the display
        plt.draw()
        plt.pause(0.001)
    
    def close(self):
        """Close the visualization window."""
        if not self.is_closed:
            plt.close(self.fig)
            self.is_closed = True


class CoblockEnvironment:
    """Manages the Minecraft-like environment."""
    
    def __init__(self, xml_file: str, visualize: bool = False):
        self.goal_graph: Dict[Tuple[int, int, int], Block] = {}
        self.current_graph: Dict[Tuple[int, int, int], Block] = {}
        self.visualizer = None
        
        # Load and parse the XML structure
        self._load_structure(xml_file)
        
        # Validate that the goal structure adheres to gravity
        if not self._validate_gravity(self.goal_graph):
            raise ValueError("Goal structure does not adhere to gravity constraints")
        
        # Initialize visualizer if requested
        if visualize:
            self.visualizer = BlockVisualizer(self)
    
    def _load_structure(self, xml_file: str):
        """Load the structure from an XML file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            if root.tag != 'Structure':
                raise ValueError("Root element must be 'Structure'")
            
            # Parse all blocks from all goals
            for goal in root.findall('Goal'):
                for block_elem in goal.findall('Block'):
                    # Parse attributes
                    color = block_elem.get('color')
                    pos_str = block_elem.get('pos')
                    
                    if not color or not pos_str:
                        raise ValueError("Block must have 'color' and 'pos' attributes")
                    
                    # Parse position tuple
                    pos = eval(pos_str)  # Safe in this controlled context
                    if not isinstance(pos, tuple) or len(pos) != 3:
                        raise ValueError(f"Position must be a 3-tuple, got {pos}")
                    
                    # Create block and add to goal graph
                    block = Block(color=color, pos=pos)
                    self.goal_graph[pos] = block
            
            # Connect neighbors in the goal graph
            self._connect_neighbors(self.goal_graph)
            
            logger.info(f"Loaded structure with {len(self.goal_graph)} blocks")
            
        except Exception as e:
            logger.error(f"Error loading XML file: {e}")
            raise
    
    def _connect_neighbors(self, graph: Dict[Tuple[int, int, int], Block]):
        """Connect blocks to their neighbors in the graph."""
        directions = {
            'up': (0, 0, 1),
            'down': (0, 0, -1),
            'north': (0, 1, 0),
            'south': (0, -1, 0),
            'east': (1, 0, 0),
            'west': (-1, 0, 0)
        }
        
        for pos, block in graph.items():
            for direction, delta in directions.items():
                neighbor_pos = (pos[0] + delta[0], pos[1] + delta[1], pos[2] + delta[2])
                if neighbor_pos in graph:
                    block.neighbors[direction] = graph[neighbor_pos]
    
    def _validate_gravity(self, graph: Dict[Tuple[int, int, int], Block]) -> bool:
        """Validate that all blocks in the graph adhere to gravity."""
        if not graph:
            return True
        
        # Find all blocks on the ground (z=0)
        ground_blocks = {pos for pos in graph if pos[2] == 0}
        
        if not ground_blocks:
            logger.error("No blocks on the ground (z=0)")
            return False
        
        # Use BFS to find all blocks connected to ground
        connected_blocks = set()
        to_visit = list(ground_blocks)
        
        while to_visit:
            current_pos = to_visit.pop(0)
            if current_pos in connected_blocks:
                continue
                
            connected_blocks.add(current_pos)
            current_block = graph[current_pos]
            
            # Add all neighbors to visit list
            for neighbor in current_block.neighbors.values():
                if neighbor and neighbor.pos not in connected_blocks:
                    to_visit.append(neighbor.pos)
        
        # Check if all blocks are connected to ground
        all_blocks = set(graph.keys())
        unconnected = all_blocks - connected_blocks
        
        if unconnected:
            logger.error(f"Blocks not connected to ground: {unconnected}")
            return False
        
        return True
    
    def _is_adjacent_to_existing(self, pos: Tuple[int, int, int]) -> bool:
        """Check if a position is adjacent to at least one existing block."""
        if not self.current_graph:
            # First block must be on the ground
            return pos[2] == 0
        
        directions = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
        
        for dx, dy, dz in directions:
            neighbor_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if neighbor_pos in self.current_graph:
                return True
        
        return False
    
    def place_block(self, color: str, pos: Tuple[int, int, int], agent: str) -> bool:
        """Place a block at the specified position."""
        # Check if position is already occupied
        if pos in self.current_graph:
            logger.warning(f"Cannot place block at {pos}: position already occupied")
            return False
        
        # Check if placement adheres to gravity
        if not self._is_adjacent_to_existing(pos):
            logger.warning(f"Cannot place block at {pos}: does not adhere to gravity")
            return False
        
        # Create and place the block
        new_block = Block(color=color, pos=pos, owner=agent)
        self.current_graph[pos] = new_block
        
        # Update neighbor connections
        self._connect_neighbors(self.current_graph)
        
        logger.info(f"Block placed by {agent} at {pos} with color {color}")
        
        # Update visualization if enabled
        if self.visualizer:
            self.visualizer.update()
        
        return True
    
    def remove_block(self, pos: Tuple[int, int, int], agent: str) -> bool:
        """Remove a block at the specified position."""
        # Check if block exists
        if pos not in self.current_graph:
            logger.warning(f"Cannot remove block at {pos}: no block at position")
            return False
        
        # Check if agent owns the block
        block = self.current_graph[pos]
        if block.owner != agent:
            logger.warning(f"Cannot remove block at {pos}: owned by {block.owner}, not {agent}")
            return False
        
        # Remove the block
        del self.current_graph[pos]
        
        # Update neighbor connections
        self._connect_neighbors(self.current_graph)
        
        # Validate that remaining structure still adheres to gravity
        if not self._validate_gravity(self.current_graph):
            # Restore the block if removal breaks gravity
            self.current_graph[pos] = block
            self._connect_neighbors(self.current_graph)
            logger.warning(f"Cannot remove block at {pos}: would break gravity constraints")
            return False
        
        logger.info(f"Block removed by {agent} at {pos}")
        
        # Update visualization if enabled
        if self.visualizer:
            self.visualizer.update()
        
        return True
    
    def get_current_state(self) -> Dict[Tuple[int, int, int], Dict[str, str]]:
        """Get the current state of the environment."""
        return {
            pos: {'color': block.color, 'owner': block.owner}
            for pos, block in self.current_graph.items()
        }
    
    def get_goal_state(self) -> Dict[Tuple[int, int, int], str]:
        """Get the goal state of the environment."""
        return {pos: block.color for pos, block in self.goal_graph.items()}
    
    def is_goal_achieved(self) -> bool:
        """Check if the current state matches the goal state."""
        goal_positions = set(self.goal_graph.keys())
        current_positions = set(self.current_graph.keys())
        
        if goal_positions != current_positions:
            return False
        
        for pos in goal_positions:
            if self.goal_graph[pos].color != self.current_graph[pos].color:
                return False
        
        return True
    
    def close(self):
        """Close the environment and any associated resources."""
        if self.visualizer:
            self.visualizer.close()


def main():
    """Simple CLI for testing the CoblockEnvironment."""
    parser = argparse.ArgumentParser(description='Test CoblockEnvironment with an XML file')
    parser.add_argument('xml_file', help='Path to the XML structure file')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Enable 3D visualization')
    args = parser.parse_args()
    
    try:
        env = CoblockEnvironment(args.xml_file, visualize=args.visualize)
        print(f"Environment loaded successfully!")
        print(f"Goal state: {env.get_goal_state()}")
        
        if args.visualize:
            print("Visualization window opened. You can rotate the view with mouse.")
            print("Wireframe blocks show the goal state, solid blocks show current state.")
        
        # Simple interactive loop for testing
        while True:
            print("\nCommands: place <color> <x,y,z> <agent> | remove <x,y,z> <agent> | state | goal | check | quit")
            cmd = input("> ").strip().split()
            
            if not cmd:
                continue
            
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'state':
                print(f"Current state: {env.get_current_state()}")
            elif cmd[0] == 'goal':
                print(f"Goal state: {env.get_goal_state()}")
            elif cmd[0] == 'check':
                print(f"Goal achieved: {env.is_goal_achieved()}")
            elif cmd[0] == 'place' and len(cmd) == 4:
                color = cmd[1]
                pos = eval(cmd[2])  # e.g., "(0,0,0)"
                agent = cmd[3]
                success = env.place_block(color, pos, agent)
                print(f"Place block: {'Success' if success else 'Failed'}")
            elif cmd[0] == 'remove' and len(cmd) == 3:
                pos = eval(cmd[1])  # e.g., "(0,0,0)"
                agent = cmd[2]
                success = env.remove_block(pos, agent)
                print(f"Remove block: {'Success' if success else 'Failed'}")
            else:
                print("Invalid command")
        
        # Clean up
        env.close()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()