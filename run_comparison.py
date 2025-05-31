#!/usr/bin/env python3
"""
Simple script to run model comparisons with default settings.
"""

import sys
import os
from pathlib import Path
from model_comparison_orchestrator import ModelComparisonOrchestrator, create_test_structures


def main():
    print("=" * 80)
    print("LLM MODEL COMPARISON RUNNER")
    print("=" * 80)
    
    # Check for API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in environment variables")
        print("Please set it using: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)
    
    # Configuration
    trials_per_pairing = 3
    max_turns = 50
    output_dir = f"comparison_results_{Path(__file__).stem}"
    
    # Create test structures
    print("\nCreating test structures...")
    structures = create_test_structures("test_structures")
    print(f"Created {len(structures)} test structures:")
    for s in structures:
        print(f"  - {os.path.basename(s)}")
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  - Models: Sonnet 4, Sonnet 3.7, Sonnet 3.5, Haiku 3.5")
    print(f"  - Total pairings: 10 (including self-pairings)")
    print(f"  - Structures: {len(structures)}")
    print(f"  - Trials per pairing: {trials_per_pairing}")
    print(f"  - Max turns per game: {max_turns}")
    print(f"  - Total runs: {10 * len(structures) * trials_per_pairing}")
    print(f"  - Output directory: {output_dir}")
    
    # Confirm
    response = input("\nProceed with comparison? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run comparisons
    print("\nStarting comparisons...")
    orchestrator = ModelComparisonOrchestrator(
        structures=structures,
        trials_per_pairing=trials_per_pairing,
        max_turns=max_turns,
        output_dir=output_dir
    )
    
    try:
        orchestrator.run_all_comparisons()
        orchestrator.generate_report()
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}/")
        print("Files generated:")
        print(f"  - Detailed log: {output_dir}/comparison_log_*.log")
        print(f"  - Raw results: {output_dir}/results.json")
        print(f"  - Report: {output_dir}/report/")
        print(f"    - Text report: report.txt")
        print(f"    - CSV data: detailed_results.csv")
        print(f"    - Visualizations: *.png")
        
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
        print("Partial results have been saved.")
    except Exception as e:
        print(f"\n\nError during comparison: {e}")
        print("Partial results may have been saved.")
        raise


if __name__ == "__main__":
    main()