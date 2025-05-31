#!/usr/bin/env python3
"""
Model Comparison Orchestrator for LLM Coordination Benchmarking.
Compares different language models in collaborative building tasks.
"""

import os
import sys
import json
import logging
import time
import datetime
import itertools
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

from llm_coordinator import LLMCoordinator


# Model configurations
MODEL_CONFIGS = {
    "sonnet-4": {
        "name": "Sonnet 4",
        "model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic"
    },
    # "sonnet-3.7": {
    #     "name": "Sonnet 3.7",
    #     "model": "claude-3-5-sonnet-20240620",
    #     "provider": "anthropic"
    # },
    # "sonnet-3.5": {
    #     "name": "Sonnet 3.5",
    #     "model": "claude-3-sonnet-20240229",
    #     "provider": "anthropic"
    # },
    "haiku-3.5": {
        "name": "Haiku 3.5",
        "model": "claude-3-haiku-20240307",
        "provider": "anthropic"
    }
}


@dataclass
class TrialResult:
    """Results from a single trial."""
    pairing: Tuple[str, str]
    structure: str
    trial_num: int
    success: bool
    turns_taken: int
    total_actions: int
    failed_actions: int
    time_taken: float
    agent1_actions: int
    agent2_actions: int
    chat_messages_sent: int
    
    @property
    def success_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return (self.total_actions - self.failed_actions) / self.total_actions


@dataclass
class ComparisonResults:
    """Aggregated results for all trials."""
    results: List[TrialResult] = field(default_factory=list)
    
    def add_result(self, result: TrialResult):
        self.results.append(result)
    
    def get_summary_by_pairing(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Get summary statistics grouped by model pairing."""
        summary = defaultdict(lambda: {
            'trials': 0,
            'successes': 0,
            'avg_turns': [],
            'avg_time': [],
            'avg_actions': [],
            'avg_failed_actions': [],
            'success_rates': []
        })
        
        for result in self.results:
            pairing = result.pairing
            summary[pairing]['trials'] += 1
            if result.success:
                summary[pairing]['successes'] += 1
                summary[pairing]['avg_turns'].append(result.turns_taken)
            summary[pairing]['avg_time'].append(result.time_taken)
            summary[pairing]['avg_actions'].append(result.total_actions)
            summary[pairing]['avg_failed_actions'].append(result.failed_actions)
            summary[pairing]['success_rates'].append(result.success_rate)
        
        # Calculate averages
        for pairing, data in summary.items():
            data['success_rate'] = data['successes'] / data['trials'] if data['trials'] > 0 else 0
            data['avg_turns'] = np.mean(data['avg_turns']) if data['avg_turns'] else float('inf')
            data['avg_time'] = np.mean(data['avg_time'])
            data['avg_actions'] = np.mean(data['avg_actions'])
            data['avg_failed_actions'] = np.mean(data['avg_failed_actions'])
            data['avg_success_rate'] = np.mean(data['success_rates'])
        
        return dict(summary)
    
    def get_summary_by_structure(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics grouped by structure."""
        summary = defaultdict(lambda: defaultdict(lambda: {
            'trials': 0,
            'successes': 0,
            'avg_turns': [],
            'avg_time': []
        }))
        
        for result in self.results:
            structure = result.structure
            pairing = result.pairing
            summary[structure][pairing]['trials'] += 1
            if result.success:
                summary[structure][pairing]['successes'] += 1
                summary[structure][pairing]['avg_turns'].append(result.turns_taken)
            summary[structure][pairing]['avg_time'].append(result.time_taken)
        
        # Calculate averages
        for structure in summary:
            for pairing in summary[structure]:
                data = summary[structure][pairing]
                data['success_rate'] = data['successes'] / data['trials'] if data['trials'] > 0 else 0
                data['avg_turns'] = np.mean(data['avg_turns']) if data['avg_turns'] else float('inf')
                data['avg_time'] = np.mean(data['avg_time'])
        
        return dict(summary)


class DualLogger:
    """Logger that writes to both file and stdout."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = []  # Clear existing handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)


class ModelComparisonOrchestrator:
    """Orchestrates model comparisons across multiple structures and trials."""
    
    def __init__(self, structures: List[str], trials_per_pairing: int = 3, 
                 max_turns: int = 100, output_dir: str = "comparison_results"):
        """
        Initialize the orchestrator.
        
        Args:
            structures: List of paths to structure XML files
            trials_per_pairing: Number of trials to run for each model pairing
            max_turns: Maximum turns per game
            output_dir: Directory to save results
        """
        self.structures = structures
        self.trials_per_pairing = trials_per_pairing
        self.max_turns = max_turns
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"comparison_log_{timestamp}.log"
        self.logger = DualLogger(str(log_file))
        self.main_logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = ComparisonResults()
        
        # Model list
        self.models = list(MODEL_CONFIGS.keys())
        
        self.main_logger.info(f"Initialized orchestrator with {len(self.models)} models and {len(structures)} structures")
    
    def get_all_pairings(self) -> List[Tuple[str, str]]:
        """Get all possible model pairings (including self-pairings)."""
        # Use combinations_with_replacement to include self-pairings
        return list(itertools.combinations_with_replacement(self.models, 2))
    
    def run_single_trial(self, model1: str, model2: str, structure: str, 
                        trial_num: int) -> TrialResult:
        """Run a single trial with two models."""
        self.main_logger.info(f"\n{'='*60}")
        self.main_logger.info(f"Running trial {trial_num} for {MODEL_CONFIGS[model1]['name']} vs {MODEL_CONFIGS[model2]['name']}")
        self.main_logger.info(f"Structure: {structure}")
        self.main_logger.info(f"{'='*60}")
        
        # Create agent configurations
        agent_configs = [
            {
                "name": "agent1",
                "model": MODEL_CONFIGS[model1]["model"],
                "provider": MODEL_CONFIGS[model1]["provider"]
            },
            {
                "name": "agent2",
                "model": MODEL_CONFIGS[model2]["model"],
                "provider": MODEL_CONFIGS[model2]["provider"]
            }
        ]
        
        # Track metrics
        start_time = time.time()
        
        try:
            # Create coordinator (without visualization for automated runs)
            coordinator = LLMCoordinator(
                environment_xml=structure,
                agent_configs=agent_configs,
                visualize=False
            )
            
            # Run the game
            success = coordinator.run_game(max_turns=self.max_turns)
            
            # Extract metrics
            turns_taken = coordinator.turn
            total_actions = coordinator.total_actions
            failed_actions = coordinator.failed_actions_count
            
            # Count actions per agent
            agent1_actions = len([a for a in coordinator.world_actions if a.agent == "agent1"])
            agent2_actions = len([a for a in coordinator.world_actions if a.agent == "agent2"])
            
            # Count chat messages
            chat_messages = sum(len(msgs) for agent in coordinator.agents.values() 
                              for msgs in agent.chat_messages.values())
            
            time_taken = time.time() - start_time
            
            result = TrialResult(
                pairing=(model1, model2),
                structure=os.path.basename(structure),
                trial_num=trial_num,
                success=success,
                turns_taken=turns_taken if success else self.max_turns,
                total_actions=total_actions,
                failed_actions=failed_actions,
                time_taken=time_taken,
                agent1_actions=agent1_actions,
                agent2_actions=agent2_actions,
                chat_messages_sent=chat_messages
            )
            
            self.main_logger.info(f"Trial completed: {'SUCCESS' if success else 'FAILURE'} in {turns_taken} turns")
            
            # Clean up
            coordinator.env.close()
            
            # Small delay between trials
            time.sleep(2)
            
            return result
            
        except Exception as e:
            self.main_logger.error(f"Error in trial: {e}")
            # Return a failed result
            return TrialResult(
                pairing=(model1, model2),
                structure=os.path.basename(structure),
                trial_num=trial_num,
                success=False,
                turns_taken=self.max_turns,
                total_actions=0,
                failed_actions=0,
                time_taken=time.time() - start_time,
                agent1_actions=0,
                agent2_actions=0,
                chat_messages_sent=0
            )
    
    def run_all_comparisons(self):
        """Run all model comparisons."""
        pairings = self.get_all_pairings()
        total_runs = len(pairings) * len(self.structures) * self.trials_per_pairing
        
        self.main_logger.info(f"\nStarting comparisons:")
        self.main_logger.info(f"- Models: {[MODEL_CONFIGS[m]['name'] for m in self.models]}")
        self.main_logger.info(f"- Pairings: {len(pairings)}")
        self.main_logger.info(f"- Structures: {len(self.structures)}")
        self.main_logger.info(f"- Trials per pairing: {self.trials_per_pairing}")
        self.main_logger.info(f"- Total runs: {total_runs}")
        
        run_count = 0
        
        for structure in self.structures:
            for model1, model2 in pairings:
                for trial_num in range(1, self.trials_per_pairing + 1):
                    run_count += 1
                    self.main_logger.info(f"\n[{run_count}/{total_runs}] Running comparison...")
                    
                    result = self.run_single_trial(model1, model2, structure, trial_num)
                    self.results.add_result(result)
                    
                    # Save intermediate results
                    self.save_results()
        
        self.main_logger.info("\nAll comparisons completed!")
    
    def save_results(self):
        """Save results to JSON file."""
        results_file = self.output_dir / "results.json"
        
        results_data = {
            'metadata': {
                'models': self.models,
                'structures': self.structures,
                'trials_per_pairing': self.trials_per_pairing,
                'max_turns': self.max_turns,
                'timestamp': datetime.datetime.now().isoformat()
            },
            'results': [asdict(r) for r in self.results.results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def generate_report(self):
        """Generate a comprehensive report with visualizations."""
        self.main_logger.info("\nGenerating report...")
        
        # Create report directory
        report_dir = self.output_dir / "report"
        report_dir.mkdir(exist_ok=True)
        
        # Generate summary statistics
        pairing_summary = self.results.get_summary_by_pairing()
        structure_summary = self.results.get_summary_by_structure()
        
        # Create visualizations
        self._create_success_rate_heatmap(pairing_summary, report_dir)
        self._create_turns_comparison(pairing_summary, report_dir)
        self._create_performance_by_structure(structure_summary, report_dir)
        self._create_action_efficiency_plot(report_dir)
        
        # Generate text report
        self._generate_text_report(pairing_summary, structure_summary, report_dir)
        
        # Generate CSV for detailed analysis
        self._generate_csv_report(report_dir)
        
        self.main_logger.info(f"Report generated in {report_dir}")
    
    def _create_success_rate_heatmap(self, pairing_summary: Dict, report_dir: Path):
        """Create a heatmap of success rates for model pairings."""
        # Create matrix for heatmap
        models = self.models
        matrix = np.zeros((len(models), len(models)))
        
        for (model1, model2), data in pairing_summary.items():
            i = models.index(model1)
            j = models.index(model2)
            success_rate = data['success_rate'] * 100
            matrix[i][j] = success_rate
            matrix[j][i] = success_rate  # Symmetric
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        model_names = [MODEL_CONFIGS[m]['name'] for m in models]
        
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                   xticklabels=model_names, yticklabels=model_names,
                   cbar_kws={'label': 'Success Rate (%)'},
                   vmin=0, vmax=100)
        
        plt.title('Model Pairing Success Rates', fontsize=16)
        plt.tight_layout()
        plt.savefig(report_dir / 'success_rate_heatmap.png', dpi=300)
        plt.close()
    
    def _create_turns_comparison(self, pairing_summary: Dict, report_dir: Path):
        """Create a bar chart comparing average turns to completion."""
        # Prepare data
        pairings = []
        avg_turns = []
        
        for (model1, model2), data in pairing_summary.items():
            if data['avg_turns'] != float('inf'):
                pairing_name = f"{MODEL_CONFIGS[model1]['name']} + {MODEL_CONFIGS[model2]['name']}"
                pairings.append(pairing_name)
                avg_turns.append(data['avg_turns'])
        
        # Sort by average turns
        sorted_indices = np.argsort(avg_turns)
        pairings = [pairings[i] for i in sorted_indices]
        avg_turns = [avg_turns[i] for i in sorted_indices]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.barh(pairings, avg_turns)
        
        # Color bars based on performance
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Average Turns to Completion', fontsize=12)
        plt.title('Model Pairing Performance (Lower is Better)', fontsize=16)
        plt.tight_layout()
        plt.savefig(report_dir / 'turns_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_by_structure(self, structure_summary: Dict, report_dir: Path):
        """Create grouped bar chart showing performance across different structures."""
        if not structure_summary:
            return
        
        structures = list(structure_summary.keys())
        model_pairs = list(next(iter(structure_summary.values())).keys())
        
        # Prepare data
        success_rates = defaultdict(list)
        for structure in structures:
            for pair in model_pairs:
                if pair in structure_summary[structure]:
                    rate = structure_summary[structure][pair]['success_rate'] * 100
                else:
                    rate = 0
                success_rates[pair].append(rate)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(structures))
        width = 0.8 / len(model_pairs)
        
        for i, (pair, rates) in enumerate(success_rates.items()):
            offset = (i - len(model_pairs)/2 + 0.5) * width
            pair_name = f"{MODEL_CONFIGS[pair[0]]['name']} + {MODEL_CONFIGS[pair[1]]['name']}"
            ax.bar(x + offset, rates, width, label=pair_name)
        
        ax.set_xlabel('Structure', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Performance by Structure and Model Pairing', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(structures)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(report_dir / 'performance_by_structure.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_action_efficiency_plot(self, report_dir: Path):
        """Create scatter plot showing action efficiency."""
        # Prepare data
        data_points = defaultdict(lambda: {'total_actions': [], 'failed_actions': []})
        
        for result in self.results.results:
            if result.success:
                pair_name = f"{MODEL_CONFIGS[result.pairing[0]]['name']} + {MODEL_CONFIGS[result.pairing[1]]['name']}"
                data_points[pair_name]['total_actions'].append(result.total_actions)
                data_points[pair_name]['failed_actions'].append(result.failed_actions)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        for pair_name, data in data_points.items():
            if data['total_actions']:
                plt.scatter(data['total_actions'], data['failed_actions'], 
                           label=pair_name, alpha=0.6, s=100)
        
        plt.xlabel('Total Actions', fontsize=12)
        plt.ylabel('Failed Actions', fontsize=12)
        plt.title('Action Efficiency by Model Pairing', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(report_dir / 'action_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, pairing_summary: Dict, structure_summary: Dict, report_dir: Path):
        """Generate detailed text report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("MODEL COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total trials: {len(self.results.results)}")
        report_lines.append(f"Models compared: {', '.join([MODEL_CONFIGS[m]['name'] for m in self.models])}")
        report_lines.append(f"Structures tested: {len(self.structures)}")
        report_lines.append(f"Trials per pairing: {self.trials_per_pairing}")
        report_lines.append("")
        
        # Overall summary
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("-" * 40)
        
        # Best performing pairing
        best_pairing = min(pairing_summary.items(), 
                          key=lambda x: x[1]['avg_turns'] if x[1]['avg_turns'] != float('inf') else 999999)
        report_lines.append(f"Best performing pairing: {MODEL_CONFIGS[best_pairing[0][0]]['name']} + {MODEL_CONFIGS[best_pairing[0][1]]['name']}")
        report_lines.append(f"  - Success rate: {best_pairing[1]['success_rate']*100:.1f}%")
        report_lines.append(f"  - Avg turns: {best_pairing[1]['avg_turns']:.1f}")
        report_lines.append("")
        
        # Detailed pairing results
        report_lines.append("DETAILED PAIRING RESULTS")
        report_lines.append("-" * 40)
        
        for (model1, model2), data in sorted(pairing_summary.items()):
            report_lines.append(f"\n{MODEL_CONFIGS[model1]['name']} + {MODEL_CONFIGS[model2]['name']}:")
            report_lines.append(f"  Success rate: {data['success_rate']*100:.1f}%")
            report_lines.append(f"  Successful trials: {data['successes']}/{data['trials']}")
            if data['avg_turns'] != float('inf'):
                report_lines.append(f"  Avg turns to completion: {data['avg_turns']:.1f}")
            report_lines.append(f"  Avg total actions: {data['avg_actions']:.1f}")
            report_lines.append(f"  Avg failed actions: {data['avg_failed_actions']:.1f}")
            report_lines.append(f"  Action success rate: {data['avg_success_rate']*100:.1f}%")
        
        # Structure-specific results
        if structure_summary:
            report_lines.append("\n\nRESULTS BY STRUCTURE")
            report_lines.append("-" * 40)
            
            for structure, pairing_data in structure_summary.items():
                report_lines.append(f"\n{structure}:")
                for (model1, model2), data in sorted(pairing_data.items()):
                    if data['trials'] > 0:
                        report_lines.append(f"  {MODEL_CONFIGS[model1]['name']} + {MODEL_CONFIGS[model2]['name']}:")
                        report_lines.append(f"    Success rate: {data['success_rate']*100:.1f}%")
                        if data['avg_turns'] != float('inf'):
                            report_lines.append(f"    Avg turns: {data['avg_turns']:.1f}")
        
        # Save report
        report_file = report_dir / 'report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also print to console
        print('\n'.join(report_lines))
    
    def _generate_csv_report(self, report_dir: Path):
        """Generate CSV file with detailed results."""
        # Convert results to DataFrame
        data = []
        for result in self.results.results:
            row = {
                'model1': MODEL_CONFIGS[result.pairing[0]]['name'],
                'model2': MODEL_CONFIGS[result.pairing[1]]['name'],
                'structure': result.structure,
                'trial': result.trial_num,
                'success': result.success,
                'turns': result.turns_taken,
                'total_actions': result.total_actions,
                'failed_actions': result.failed_actions,
                'success_rate': result.success_rate,
                'time_seconds': result.time_taken,
                'agent1_actions': result.agent1_actions,
                'agent2_actions': result.agent2_actions,
                'chat_messages': result.chat_messages_sent
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_file = report_dir / 'detailed_results.csv'
        df.to_csv(csv_file, index=False)
        
        self.main_logger.info(f"Saved detailed results to {csv_file}")


def create_test_structures(output_dir: str = "test_structures") -> List[str]:
    """Create a set of test structures with varying complexity."""
    os.makedirs(output_dir, exist_ok=True)
    structures = []
    
    # Simple structure
    simple = f"{output_dir}/simple_structure.xml"
    with open(simple, 'w') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<Structure>
    <Goal>
        <Block color="red" pos="(0, 0, 0)"/>
        <Block color="red" pos="(1, 0, 0)"/>
    </Goal>
    <Goal>
        <Block color="blue" pos="(0, 1, 0)"/>
        <Block color="blue" pos="(1, 1, 0)"/>
    </Goal>
</Structure>''')
    structures.append(simple)
    
    # Medium structure
    medium = f"{output_dir}/medium_structure.xml"
    with open(medium, 'w') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<Structure>
    <Goal>
        <Block color="yellow" pos="(0, 0, 0)"/>
        <Block color="yellow" pos="(0, 1, 0)"/>
        <Block color="yellow" pos="(0, 0, 1)"/>
    </Goal>
    <Goal>
        <Block color="green" pos="(1, 0, 0)"/>
        <Block color="green" pos="(1, 1, 0)"/>
        <Block color="green" pos="(1, 0, 1)"/>
    </Goal>
</Structure>''')
    structures.append(medium)
    
    # Complex structure
    complex_struct = f"{output_dir}/complex_structure.xml"
    with open(complex_struct, 'w') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<Structure>
    <Goal>
        <Block color="red" pos="(0, 0, 0)"/>
        <Block color="red" pos="(1, 0, 0)"/>
        <Block color="red" pos="(0, 1, 0)"/>
        <Block color="red" pos="(1, 1, 0)"/>
    </Goal>
    <Goal>
        <Block color="blue" pos="(2, 0, 0)"/>
        <Block color="blue" pos="(2, 1, 0)"/>
        <Block color="blue" pos="(2, 0, 1)"/>
        <Block color="blue" pos="(2, 1, 1)"/>
    </Goal>
</Structure>''')
    structures.append(complex_struct)
    
    return structures


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run model comparison benchmarks')
    parser.add_argument('--structures', nargs='+', help='Paths to structure XML files')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials per pairing')
    parser.add_argument('--max-turns', type=int, default=100, help='Maximum turns per game')
    parser.add_argument('--output-dir', default='comparison_results', help='Output directory')
    parser.add_argument('--create-test-structures', action='store_true', 
                       help='Create test structures and exit')
    
    args = parser.parse_args()
    
    if args.create_test_structures:
        structures = create_test_structures()
        print(f"Created test structures: {structures}")
        return
    
    # Use provided structures or create test ones
    if args.structures:
        structures = args.structures
    else:
        print("No structures provided, creating test structures...")
        structures = create_test_structures()
    
    # Run comparisons
    orchestrator = ModelComparisonOrchestrator(
        structures=structures,
        trials_per_pairing=args.trials,
        max_turns=args.max_turns,
        output_dir=args.output_dir
    )
    
    orchestrator.run_all_comparisons()
    orchestrator.generate_report()


if __name__ == "__main__":
    main()