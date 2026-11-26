"""Command-line entry point for the bundled demo and CSV experiments."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.datasets import load_wine
from .agents import LLMAgents
from .workflow import run


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible tabular ML experiment.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Use the bundled wine classification dataset")
    csv = commands.add_parser("run", help="Use your own CSV")
    csv.add_argument("csv", type=Path)
    csv.add_argument("--target", required=True)
    csv.add_argument("--task", choices=["classification", "regression"], required=True)
    csv.add_argument("--goal", required=True)
    for command in (demo, csv):
        command.add_argument("--database", type=Path, default=Path("runs/ledger.sqlite"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--llm-model", help="Enable OpenAI planner and reviewer with this model ID")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            frame = load_wine(as_frame=True).frame
            target, task, goal = "target", "classification", "Classify wine cultivars from chemical measurements."
        else:
            frame = pd.read_csv(args.csv)
            target, task, goal = args.target, args.task, args.goal
        agents = LLMAgents(args.llm_model) if args.llm_model else None
        report = run(frame, target=target, task=task, goal=goal, database=args.database,
                     seed=args.seed, agents=agents)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
