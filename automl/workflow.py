"""Run a bounded experiment with two decision-making agents and deterministic tools."""
from typing import TypedDict
from uuid import uuid4
from langgraph.graph import END, START, StateGraph
from .agents import LocalAgents
from .experiment import Experiment
from .memory import Ledger


class State(TypedDict, total=False):
    goal: str
    context: dict
    first: dict
    review: dict
    trials: list
    winner: str
    final_score: float
    report: dict


def run(frame, *, target, task, goal, database, agents=None, seed=42):
    goal = goal.strip()
    if not goal:
        raise ValueError("Describe the goal of the experiment.")
    experiment = Experiment(frame, target, task, seed)
    ledger = Ledger(database)
    agents = LocalAgents() if agents is None else agents

    def brief(state):
        return {"context": {"goal": state["goal"], "task": task, "metric": experiment.metric,
                            "training_rows": len(experiment.X), "numeric_features": len(experiment.numeric),
                            "categorical_features": len(experiment.category),
                            "prior_runs": ledger.recall(goal, task)}}

    def plan(state):
        return {"first": agents.plan(state["context"]).model_dump()}

    def benchmark(state):
        return {"trials": [experiment.validate("baseline"), experiment.validate(state["first"]["candidate"])]}

    def review(state):
        evidence = {**state["context"], "trials": state["trials"]}
        return {"review": agents.review(evidence).model_dump()}

    def challenge(state):
        alternative = "forest" if state["first"]["candidate"] == "linear" else "linear"
        return {"trials": [*state["trials"], experiment.validate(alternative)]}

    def select(state):
        # The baseline wins ties. No holdout scores are available during selection.
        winner = max(state["trials"], key=lambda trial: trial["mean"])["name"]
        return {"winner": winner, "final_score": experiment.evaluate_final(winner)}

    def record(state):
        report = {"id": uuid4().hex, "goal": goal, "task": task, "target": target,
                  "mode": agents.mode, "seed": seed, "metric": experiment.metric,
                  "training_rows": len(experiment.X), "holdout_rows": len(experiment.truth),
                  "prior_run_ids": [r["id"] for r in state["context"]["prior_runs"]],
                  "planner": state["first"], "reviewer": state["review"], "trials": state["trials"],
                  "winner": state["winner"], "holdout_score": state["final_score"]}
        ledger.remember(report)
        return {"report": report}

    graph = StateGraph(State)
    for name, node in [("brief", brief), ("plan", plan), ("benchmark", benchmark),
                       ("assess", review), ("challenge", challenge), ("select", select), ("record", record)]:
        graph.add_node(name, node)
    graph.add_edge(START, "brief")
    graph.add_edge("brief", "plan")
    graph.add_edge("plan", "benchmark")
    graph.add_edge("benchmark", "assess")
    graph.add_conditional_edges("assess", lambda s: "challenge" if s["review"]["try_challenger"] else "select",
                                {"challenge": "challenge", "select": "select"})
    graph.add_edge("challenge", "select")
    graph.add_edge("select", "record")
    graph.add_edge("record", END)
    return graph.compile().invoke({"goal": goal})["report"]
