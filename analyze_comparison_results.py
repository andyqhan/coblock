#!/usr/bin/env python3
"""
Script to analyze and visualize existing comparison results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse


class ResultsAnalyzer:
    """Analyze and visualize model comparison results."""
    
    def __init__(self, results_file: str):
        """Load results from JSON file."""
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.metadata = self.data['metadata']
        self.results = self.data['results']
        self.df = pd.DataFrame(self.results)
        
        print(f"Loaded {len(self.results)} results from {results_file}")
        print(f"Models: {self.metadata['models']}")
        print(f"Structures: {self.metadata['structures']}")
    
    def print_summary_statistics(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Overall success rate
        success_rate = self.df['success'].mean() * 100
        print(f"\nOverall success rate: {success_rate:.1f}%")
        
        # Success rate by pairing
        print("\nSuccess rate by model pairing:")
        pairing_success = self.df.groupby(['pairing'])['success'].agg(['mean', 'count'])
        pairing_success['mean'] *= 100
        pairing_success.columns = ['Success Rate (%)', 'Trials']
        print(pairing_success.sort_values('Success Rate (%)', ascending=False))
        
        # Average turns for successful runs
        successful_df = self.df[self.df['success']]
        if not successful_df.empty:
            print(f"\nAverage turns for successful runs: {successful_df['turns_taken'].mean():.1f}")
            
            print("\nAverage turns by pairing (successful runs only):")
            turns_by_pairing = successful_df.groupby('pairing')['turns_taken'].mean().sort_values()
            for pairing, turns in turns_by_pairing.items():
                print(f"  {pairing}: {turns:.1f}")
        
        # Action efficiency
        print(f"\nOverall action success rate: {(1 - self.df['failed_actions'].sum() / self.df['total_actions'].sum()) * 100:.1f}%")
        
        # Communication patterns
        print(f"\nAverage chat messages per game: {self.df['chat_messages_sent'].mean():.1f}")
    
    def create_performance_matrix(self, output_file: str = "performance_matrix.png"):
        """Create a performance matrix visualization."""
        # Extract unique models
        models = set()
        for pairing in self.df['pairing']:
            models.update(eval(pairing))
        models = sorted(list(models))
        
        # Create matrix for success rate
        matrix = pd.DataFrame(index=models, columns=models, dtype=float)
        
        for _, row in self.df.iterrows():
            model1, model2 = eval(row['pairing'])
            if pd.isna(matrix.loc[model1, model2]):
                matrix.loc[model1, model2] = 0
                matrix.loc[model2, model1] = 0
            
            if row['success']:
                matrix.loc[model1, model2] += 1
                matrix.loc[model2, model1] += 1
        
        # Calculate success rates
        for _, row in self.df.iterrows():
            model1, model2 = eval(row['pairing'])
            total = len(self.df[(self.df['pairing'] == str((model1, model2))) | 
                               (self.df['pairing'] == str((model2, model1)))])
            if total > 0:
                matrix.loc[model1, model2] = (matrix.loc[model1, model2] / total) * 100
                matrix.loc[model2, model1] = matrix.loc[model1, model2]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Success Rate (%)'},
                   vmin=0, vmax=100)
        plt.title('Model Pairing Success Rates', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"\nSaved performance matrix to {output_file}")
    
    def create_turn_distribution(self, output_file: str = "turn_distribution.png"):
        """Create turn distribution visualization."""
        successful_df = self.df[self.df['success']]
        
        if successful_df.empty:
            print("No successful runs to visualize.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Box plot by pairing
        pairings = successful_df['pairing'].unique()
        data_by_pairing = [successful_df[successful_df['pairing'] == p]['turns_taken'].values 
                          for p in pairings]
        
        ax1.boxplot(data_by_pairing, labels=[str(p) for p in pairings])
        ax1.set_xticklabels([str(p) for p in pairings], rotation=45, ha='right')
        ax1.set_ylabel('Turns to Completion')
        ax1.set_title('Turn Distribution by Model Pairing')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of all turns
        ax2.hist(successful_df['turns_taken'], bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Turns to Completion')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Overall Turn Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_turns = successful_df['turns_taken'].mean()
        median_turns = successful_df['turns_taken'].median()
        ax2.axvline(mean_turns, color='red', linestyle='--', label=f'Mean: {mean_turns:.1f}')
        ax2.axvline(median_turns, color='green', linestyle='--', label=f'Median: {median_turns:.1f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved turn distribution to {output_file}")
    
    def create_efficiency_analysis(self, output_file: str = "efficiency_analysis.png"):
        """Create efficiency analysis visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actions vs Turns scatter
        successful_df = self.df[self.df['success']]
        if not successful_df.empty:
            for pairing in successful_df['pairing'].unique():
                pairing_df = successful_df[successful_df['pairing'] == pairing]
                ax1.scatter(pairing_df['turns_taken'], pairing_df['total_actions'], 
                           label=str(pairing), alpha=0.6)
            ax1.set_xlabel('Turns to Completion')
            ax1.set_ylabel('Total Actions')
            ax1.set_title('Actions vs Turns')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
        
        # 2. Failed actions ratio
        self.df['failed_ratio'] = self.df['failed_actions'] / self.df['total_actions']
        pairing_failed = self.df.groupby('pairing')['failed_ratio'].mean().sort_values()
        
        ax2.barh(range(len(pairing_failed)), pairing_failed.values)
        ax2.set_yticks(range(len(pairing_failed)))
        ax2.set_yticklabels([str(p) for p in pairing_failed.index])
        ax2.set_xlabel('Average Failed Action Ratio')
        ax2.set_title('Action Failure Rate by Pairing')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Communication patterns
        pairing_chat = self.df.groupby('pairing')['chat_messages_sent'].mean().sort_values(ascending=False)
        
        ax3.bar(range(len(pairing_chat)), pairing_chat.values)
        ax3.set_xticks(range(len(pairing_chat)))
        ax3.set_xticklabels([str(p) for p in pairing_chat.index], rotation=45, ha='right')
        ax3.set_ylabel('Average Chat Messages')
        ax3.set_title('Communication Frequency by Pairing')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Time efficiency
        if successful_df.empty:
            ax4.text(0.5, 0.5, 'No successful runs', ha='center', va='center')
        else:
            time_per_turn = successful_df['time_taken'] / successful_df['turns_taken']
            pairing_time = successful_df.groupby('pairing').apply(
                lambda x: (x['time_taken'] / x['turns_taken']).mean()
            ).sort_values()
            
            ax4.barh(range(len(pairing_time)), pairing_time.values)
            ax4.set_yticks(range(len(pairing_time)))
            ax4.set_yticklabels([str(p) for p in pairing_time.index])
            ax4.set_xlabel('Average Time per Turn (seconds)')
            ax4.set_title('Time Efficiency by Pairing')
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved efficiency analysis to {output_file}")
    
    def create_structure_comparison(self, output_file: str = "structure_comparison.png"):
        """Create structure comparison visualization."""
        structures = self.df['structure'].unique()
        
        if len(structures) == 1:
            print("Only one structure in results, skipping structure comparison.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(['success', 'turns_taken', 'total_actions', 'chat_messages_sent']):
            ax = axes[i]
            
            if metric == 'success':
                # Success rate by structure
                data = self.df.groupby(['structure', 'pairing'])[metric].mean() * 100
                data = data.unstack()
                data.plot(kind='bar', ax=ax)
                ax.set_ylabel('Success Rate (%)')
                ax.set_title('Success Rate by Structure')
            elif metric == 'turns_taken':
                # Only successful runs
                successful_df = self.df[self.df['success']]
                if not successful_df.empty:
                    data = successful_df.groupby(['structure', 'pairing'])[metric].mean()
                    data = data.unstack()
                    data.plot(kind='bar', ax=ax)
                    ax.set_ylabel('Average Turns')
                    ax.set_title('Turns to Completion by Structure')
            else:
                data = self.df.groupby(['structure', 'pairing'])[metric].mean()
                data = data.unstack()
                data.plot(kind='bar', ax=ax)
                ax.set_ylabel(f'Average {metric.replace("_", " ").title()}')
                ax.set_title(f'{metric.replace("_", " ").title()} by Structure')
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved structure comparison to {output_file}")
    
    def export_detailed_stats(self, output_file: str = "detailed_stats.csv"):
        """Export detailed statistics to CSV."""
        # Calculate additional metrics
        self.df['success_rate'] = self.df.apply(
            lambda x: (x['total_actions'] - x['failed_actions']) / x['total_actions'] 
            if x['total_actions'] > 0 else 0, axis=1
        )
        
        # Group by pairing and structure
        grouped = self.df.groupby(['pairing', 'structure']).agg({
            'success': ['sum', 'count', 'mean'],
            'turns_taken': ['mean', 'std', 'min', 'max'],
            'total_actions': ['mean', 'sum'],
            'failed_actions': ['mean', 'sum'],
            'chat_messages_sent': ['mean', 'sum'],
            'time_taken': ['mean', 'sum']
        }).round(2)
        
        grouped.to_csv(output_file)
        print(f"\nExported detailed statistics to {output_file}")
    
    def generate_full_analysis(self, output_dir: str = "analysis_output"):
        """Generate complete analysis with all visualizations."""
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\nGenerating full analysis in {output_dir}/")
        
        # Summary statistics
        self.print_summary_statistics()
        
        # Visualizations
        self.create_performance_matrix(f"{output_dir}/performance_matrix.png")
        self.create_turn_distribution(f"{output_dir}/turn_distribution.png")
        self.create_efficiency_analysis(f"{output_dir}/efficiency_analysis.png")
        self.create_structure_comparison(f"{output_dir}/structure_comparison.png")
        
        # Export detailed stats
        self.export_detailed_stats(f"{output_dir}/detailed_stats.csv")
        
        # Create summary report
        self._create_summary_report(f"{output_dir}/analysis_summary.txt")
        
        print(f"\nAnalysis complete! Results saved to {output_dir}/")
    
    def _create_summary_report(self, output_file: str):
        """Create a text summary report."""
        with open(output_file, 'w') as f:
            f.write("MODEL COMPARISON ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Metadata
            f.write("Dataset Information:\n")
            f.write(f"- Total trials: {len(self.results)}\n")
            f.write(f"- Models: {', '.join(self.metadata['models'])}\n")
            f.write(f"- Structures: {len(self.metadata['structures'])}\n")
            f.write(f"- Trials per pairing: {self.metadata['trials_per_pairing']}\n\n")
            
            # Key findings
            f.write("Key Findings:\n")
            
            # Best performing pairing
            pairing_success = self.df.groupby('pairing')['success'].mean()
            best_pairing = pairing_success.idxmax()
            f.write(f"- Best performing pairing: {best_pairing} ({pairing_success[best_pairing]*100:.1f}% success rate)\n")
            
            # Fastest pairing
            successful_df = self.df[self.df['success']]
            if not successful_df.empty:
                pairing_speed = successful_df.groupby('pairing')['turns_taken'].mean()
                fastest_pairing = pairing_speed.idxmin()
                f.write(f"- Fastest pairing: {fastest_pairing} ({pairing_speed[fastest_pairing]:.1f} avg turns)\n")
            
            # Most efficient pairing
            self.df['efficiency'] = (self.df['total_actions'] - self.df['failed_actions']) / self.df['total_actions']
            pairing_efficiency = self.df.groupby('pairing')['efficiency'].mean()
            most_efficient = pairing_efficiency.idxmax()
            f.write(f"- Most efficient pairing: {most_efficient} ({pairing_efficiency[most_efficient]*100:.1f}% action success rate)\n")
            
            # Communication insights
            pairing_chat = self.df.groupby('pairing')['chat_messages_sent'].mean()
            most_chatty = pairing_chat.idxmax()
            f.write(f"- Most communicative pairing: {most_chatty} ({pairing_chat[most_chatty]:.1f} avg messages)\n")
        
        print(f"Created summary report: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze model comparison results')
    parser.add_argument('results_file', help='Path to results.json file')
    parser.add_argument('--output-dir', default='analysis_output', 
                       help='Directory for analysis output')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary statistics')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_file)
    
    if args.summary_only:
        analyzer.print_summary_statistics()
    else:
        analyzer.generate_full_analysis(args.output_dir)


if __name__ == "__main__":
    main()