from types import SimpleNamespace
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.datasets import load_wine
from automl.agents import Choice, LLMAgents, LocalAgents, Review
from automl.workflow import run


def test_classifier_reuses_completed_runs(tmp_path):
    frame = load_wine(as_frame=True).frame
    kw = dict(target="target", task="classification", goal="Identify the wine cultivar", database=tmp_path/"log")
    first = run(frame, **kw)
    second = run(frame, **kw)
    assert first["prior_run_ids"] == []
    assert second["prior_run_ids"] == [first["id"]]
    assert first["training_rows"] + first["holdout_rows"] == len(frame)
    assert first["metric"] == "balanced_accuracy"
    assert first["winner"] == max(first["trials"], key=lambda t: t["mean"])["name"]
    assert 0 <= first["holdout_score"] <= 1
    assert len(first["trials"]) == 3
    assert all(len(t["folds"]) == 3 for t in first["trials"])


class StopEarly(LocalAgents):
    def review(self, context):
        assert "holdout_score" not in context
        assert len(context["trials"]) == 2
        return Review(try_challenger=False, reason="End this trial.")


def test_regression_can_finish_early(tmp_path):
    frame = pd.DataFrame({"x": range(100), "site": ["a", None, "b", "c"] * 25,
                          "target": [2*x for x in range(100)]})
    report = run(frame, target="target", task="regression", goal="Estimate output",
                 database=tmp_path/"log", agents=StopEarly())
    assert report["metric"] == "neg_mean_absolute_error"
    assert report["holdout_score"] <= 0
    assert [trial["name"] for trial in report["trials"]] == ["baseline", "linear"]


def test_rejects_unsupported_action():
    with pytest.raises(ValidationError):
        Choice(candidate="execute_python", reason="unavailable")


def test_rejects_too_little_class_support(tmp_path):
    frame = pd.DataFrame({"x": range(25), "target": [1] + [0]*24})
    with pytest.raises(ValueError, match="four rows"):
        run(frame, target="target", task="classification", goal="classify", database=tmp_path/"log")


def test_llm_tool_response_contract():
    def create(**request):
        name = request["tool_choice"]["function"]["name"]
        assert request["tools"][0]["function"]["name"] == name
        arguments = '{"candidate":"forest","reason":"Test a nonlinear model."}'
        call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])
    agents = LLMAgents.__new__(LLMAgents)
    agents.model = "test-model"
    agents.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert agents.plan({"task": "classification"}).candidate == "forest"
