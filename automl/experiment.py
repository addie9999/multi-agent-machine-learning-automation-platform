"""Fold-local preprocessing and a reserved final evaluation set."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Experiment:
    def __init__(self, frame, target, task, seed):
        if task not in {"classification", "regression"} or not frame.columns.is_unique:
            raise ValueError("Choose classification or regression and use unique column names.")
        if target not in frame or len(frame.columns) < 2:
            raise ValueError("Supply a target column and at least one predictor.")
        frame = frame.dropna(subset=[target])
        if len(frame) < 20:
            raise ValueError("At least 20 rows with a known target are required.")
        self.task, self.seed = task, seed
        self.classify = task == "classification"
        y = frame[target]
        if self.classify:
            y = y.astype(str)
            if y.nunique() < 2 or y.value_counts().min() < 4:
                raise ValueError("Classification requires two or more classes and four rows per class.")
        else:
            y = pd.to_numeric(y, errors="raise")
            if not np.isfinite(y).all():
                raise ValueError("Regression targets must be finite.")
        X = frame.drop(columns=[target]).copy()
        self.numeric = X.select_dtypes(include="number").columns.tolist()
        self.category = [c for c in X.columns if c not in self.numeric]
        X[self.numeric] = X[self.numeric].replace([np.inf, -np.inf], np.nan)
        for c in self.category:
            X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        test_size = max(int(np.ceil(len(y) * 0.2)), y.nunique() if self.classify else 1)
        self.X, self.holdout, self.y, self.truth = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y if self.classify else None)
        self.metric = "balanced_accuracy" if self.classify else "neg_mean_absolute_error"
        splitter = StratifiedKFold if self.classify else KFold
        self.folds = list(splitter(n_splits=3, shuffle=True, random_state=seed).split(self.X, self.y))

    def estimator(self, name):
        if name == "baseline":
            model = DummyClassifier(strategy="prior") if self.classify else DummyRegressor(strategy="median")
        elif name == "linear":
            model = LogisticRegression(max_iter=1500) if self.classify else Ridge(solver="lsqr")
        elif name == "forest":
            forest = RandomForestClassifier if self.classify else RandomForestRegressor
            model = forest(n_estimators=64, min_samples_leaf=2, random_state=self.seed, n_jobs=1)
        else:
            raise ValueError(f"Unknown model family: {name}")
        numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        categorical = make_pipeline(SimpleImputer(strategy="constant", fill_value="(missing)", keep_empty_features=True),
                                    OneHotEncoder(handle_unknown="ignore"))
        return make_pipeline(ColumnTransformer([("numbers", numeric, self.numeric),
                                                ("categories", categorical, self.category)]), model)

    def validate(self, name):
        scores = cross_val_score(self.estimator(name), self.X, self.y, cv=self.folds,
                                 scoring=self.metric, error_score="raise")
        return {"name": name, "mean": float(scores.mean()), "folds": scores.tolist()}

    def evaluate_final(self, name):
        model = self.estimator(name).fit(self.X, self.y)
        return float(get_scorer(self.metric)(model, self.holdout, self.truth))
