#!/usr/bin/env python3
"""
run_comparison.py: Script to run model comparisons with configurable settings.
"""

import sys
import os
import argparse
from pathlib import Path
from itertools import combinations_with_replacement
from model_comparison_orchestrator import ModelComparisonOrchestrator, create_test_structures, MODEL_CONFIGS


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run LLM model comparison benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default test structures
  python run_comparison.py

  # Run with custom structures
  python run_comparison.py --structures path/to/struct1.xml path/to/struct2.xml

  # Run with custom settings
  python run_comparison.py --trials 5 --max-turns 100 --output-dir my_results

  # Create test structures only
  python run_comparison.py --create-test-structures
        """
    )

    parser.add_argument('--structures', nargs='+', help='Paths to structure XML files')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials per pairing')
    parser.add_argument('--max-turns', type=int, default=40, help='Maximum turns per game')
    parser.add_argument('--output-dir', default='comparison_results', help='Output directory')
    parser.add_argument('--create-test-structures', action='store_true',
                       help='Create test structures and exit')

    args = parser.parse_args()

    # If only creating test structures
    if args.create_test_structures:
        print("Creating test structures...")
        structures = create_test_structures("test_structures")
        print(f"Created {len(structures)} test structures:")
        for s in structures:
            print(f"  - {s}")
        return

    print("=" * 80)
    print("LLM MODEL COMPARISON RUNNER")
    print("=" * 80)

    # Check for API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in environment variables")
        print("Please set it using: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    # Get structures
    if args.structures:
        structures = args.structures
        print(f"\nUsing provided structures:")
        for s in structures:
            print(f"  - {s}")
    else:
        print("\nNo structures provided, creating test structures...")
        structures = create_test_structures("test_structures")
        print(f"Created {len(structures)} test structures:")
        for s in structures:
            print(f"  - {os.path.basename(s)}")

    # Validate structures exist
    for structure in structures:
        if not os.path.exists(structure):
            print(f"ERROR: Structure file not found: {structure}")
            sys.exit(1)

    # Calculate total runs
    num_pairings = len(list(combinations_with_replacement(MODEL_CONFIGS.keys(), 2)))
    total_runs = num_pairings * len(structures) * args.trials

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  - Models: {list(MODEL_CONFIGS.keys())}")
    print(f"  - Total pairings: {num_pairings} (including self-pairings)")
    print(f"  - Structures: {len(structures)}")
    print(f"  - Trials per pairing: {args.trials}")
    print(f"  - Max turns per game: {args.max_turns}")
    print(f"  - Total runs: {total_runs}")
    print(f"  - Output directory: {args.output_dir}")

    # Confirm for interactive runs
    if sys.stdin.isatty():  # Only prompt if running interactively
        response = input("\nProceed with comparison? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    else:
        print("\nStarting comparisons...")

    # Run comparisons
    orchestrator = ModelComparisonOrchestrator(
        structures=structures,
        trials_per_pairing=args.trials,
        max_turns=args.max_turns,
        output_dir=args.output_dir
    )

    try:
        orchestrator.run_all_comparisons()
        orchestrator.generate_report()

        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {args.output_dir}/")
        print("Files generated:")
        print(f"  - Detailed log: {args.output_dir}/comparison_log_*.log")
        print(f"  - Raw results: {args.output_dir}/results.json")
        print(f"  - Report: {args.output_dir}/report/")
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
