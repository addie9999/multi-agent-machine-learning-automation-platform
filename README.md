# Multi-Agent Machine Learning Automation Platform

Give the platform a CSV, a target column, and a modelling goal. A planner picks a first model family. A reviewer reads cross-validation results and decides whether to try a challenger. The final report records both decisions, every fold score, and the winning model's score on a reserved test set.

The experiment runs locally. Use the deterministic offline policy for a reproducible demo, or enable two LLM decision roles with an OpenAI model that supports tool calls.

## Try it in two minutes

With Python 3.11 or later, run these commands from the root of this repository in a virtual environment:

```sh
python -m pip install -e ".[dev]"
python -m automl.cli demo
python -m pytest -q
```

The demo classifies wine cultivars using scikit-learn's bundled chemical measurements. It requires no API key or dataset download. On one offline run with seed 42 and scikit-learn 1.7.2, the model comparison was:

| Model | Mean balanced accuracy across three validation folds |
| --- | ---: |
| Majority-class baseline | 0.3333 |
| Logistic regression | 0.9766 |
| Random forest | 0.9488 |

The graph selected logistic regression using 142 development rows and scored it once on 36 held-out rows. Its holdout balanced accuracy was **0.9667**. These figures describe this dataset and split. Repeat the demo with the same database to see the previous experiment's ID in `prior_run_ids`.

## Run it on a CSV

```sh
ml-experiment run orders.csv --target delivery_days --task regression --goal "Estimate delivery time from order details"
```

Choose `classification` or `regression` explicitly. Classification uses balanced accuracy. Regression uses negative mean absolute error, so a value closer to zero is better. Drop identifier columns and any features that would be unavailable when making the prediction. The split assumes independent rows. Time series and repeated observations from the same subject require a different evaluation design.

## What the agents control

1. **Brief.** Split the data, summarize the training partition, and retrieve up to three relevant earlier experiments from the local SQLite ledger. Retrieval uses character n-gram TF-IDF over goals within the same task.
2. **Plan.** The planner chooses a linear model or a random forest and records a reason.
3. **Benchmark.** Evaluate the first candidate and a dummy baseline on identical folds. Numeric imputation, scaling, and category encoding are fitted inside each fold.
4. **Review.** The reviewer receives the goal, dataset summary, validation scores, and retrieved context. It can stop or request one additional trial.
5. **Challenge.** When requested, run the other model family on the same folds.
6. **Select.** Choose the highest mean validation score, including the baseline. Refit the winner and evaluate it on the reserved partition.
7. **Record.** Write the completed experiment to SQLite and emit a JSON report. The ledger stores the goal, target name, decisions, scores, and run identifiers. The source rows stay outside the ledger.

### Enable LLM tool calling

Install the optional client with `python -m pip install -e ".[llm]"`. Set `OPENAI_API_KEY` in your environment, then append `--llm-model MODEL_ID` to either CLI command, substituting a tool-capable model available to your account. The planner and reviewer request one schema-validated action each. Model fitting, scoring, branching limits, and winner selection run in Python. Provider requests include the goal, feature counts, relevant past goal summaries, and validation results. The implementation sends no CSV rows.

Use `--database PATH` to choose the ledger and `--seed N` to change the split. The default ledger is `runs/ledger.sqlite`. Each successful run adds one experiment. LLM calls require provider access and can incur usage charges.

## Read the implementation

| File | Responsibility |
| --- | --- |
| [workflow.py](automl/workflow.py) | LangGraph state, conditional routing, evidence passed between agents |
| [agents.py](automl/agents.py) | Planner and reviewer roles, offline policies, validated LLM tool calls |
| [experiment.py](automl/experiment.py) | Preprocessing, comparable folds, model families, holdout evaluation |
| [memory.py](automl/memory.py) | SQLite run history and text-based retrieval |
| [test_platform.py](tests/test_platform.py) | Both review branches, classification and regression, retrieval, invalid actions, simulated provider response |

## License

[MIT License](LICENSE) · Copyright 2026 Aditya Roy.
