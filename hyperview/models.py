"""One pipeline (feature selector -> regressor) per soil parameter."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hyperview.config import Config
from hyperview.selection import make_selector

logger = logging.getLogger(__name__)

PARAM_SPACES = {
    "rf": {
        "model__n_estimators": [200, 300, 500, 700, 1000],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.3, None],
    },
    "xgb": {
        "model__n_estimators": [200, 400, 700, 1000],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 6],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 1.0],
        "model__min_child_weight": [1, 5, 10],
        "model__reg_alpha": [0, 0.1, 1],
        "model__reg_lambda": [1, 5, 10],
    },
    "knn": {
        "model__knn__n_neighbors": [3, 5, 7, 9, 11, 15, 20, 30],
        "model__knn__weights": ["uniform", "distance"],
        "model__knn__p": [1, 2],
    },
}


def make_regressor(cfg: Config, n_jobs: int = 1):
    if cfg.model_type == "rf":
        return RandomForestRegressor(random_state=cfg.random_seed, n_jobs=n_jobs)
    if cfg.model_type == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=cfg.random_seed, n_jobs=n_jobs, tree_method="hist")
    if cfg.model_type == "knn":
        # distance-based, so features are standardized first (originally a pH-only model by @KamiLMusiol)
        return Pipeline([("scale", StandardScaler()), ("knn", KNeighborsRegressor(n_jobs=n_jobs))])
    raise ValueError(f"Unknown model type '{cfg.model_type}', use 'rf', 'xgb' or 'knn'")


def make_pipeline(cfg: Config, params: dict | None = None, n_jobs: int = 1) -> Pipeline:
    pipeline = Pipeline([("select", make_selector(cfg, n_jobs)), ("model", make_regressor(cfg, n_jobs))])
    if params:
        pipeline.set_params(**params)
    return pipeline


class MeanBaseline:
    """The challenge baseline: predicts the training mean of every target."""

    def fit(self, X, Y):
        self.mean_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1))


class SoilModel:
    """Four independent pipelines, one per target, each with its own selected features."""

    def __init__(self, cfg: Config, params: dict[str, dict] | None = None, n_jobs: int | None = None):
        self.cfg = cfg
        self.params = params or {}
        self.n_jobs = cfg.n_jobs if n_jobs is None else n_jobs
        self.pipelines: dict[str, Pipeline] = {}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SoilModel":
        for i, target in enumerate(self.cfg.targets):
            pipeline = make_pipeline(self.cfg, self.params.get(target), self.n_jobs)
            self.pipelines[target] = pipeline.fit(X, Y[:, i])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.pipelines:
            raise RuntimeError("Fit or load the model first")
        return np.column_stack([self.pipelines[t].predict(X) for t in self.cfg.targets])

    # settings a loaded model needs to rebuild exactly the same features
    SETTINGS = ("gauss_sigma_xy", "sg_window", "sg_poly", "feature_groups", "model_type", "selector", "n_features")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        settings = {name: getattr(self.cfg, name) for name in self.SETTINGS}
        joblib.dump({"pipelines": self.pipelines, "params": self.params, "settings": settings}, path, compress=3)

    @classmethod
    def load(cls, cfg: Config, path: Path) -> "SoilModel":
        """Load a saved model; its feature settings override the ones in ``cfg``."""
        state = joblib.load(path)
        model = cls(replace(cfg, **state.get("settings", {})), state["params"])
        model.pipelines = state["pipelines"]
        return model

    def selected_features(self, names: list[str]) -> dict[str, list[str]]:
        """Names of the features each target's selector kept."""
        return {t: [names[i] for i in p.named_steps["select"].get_support(indices=True)]
                for t, p in self.pipelines.items()}


def tune_hyperparameters(cfg: Config, X: np.ndarray, Y: np.ndarray) -> dict[str, dict]:
    """Randomized search per target, optimizing MSE. Selection is re-fitted in every fold."""
    best: dict[str, dict] = {}
    cv = KFold(n_splits=cfg.tune_folds, shuffle=True, random_state=cfg.random_seed)
    for i, target in enumerate(cfg.targets):
        search = RandomizedSearchCV(make_pipeline(cfg, n_jobs=1), PARAM_SPACES[cfg.model_type],
                                    n_iter=cfg.tune_iter, scoring=cfg.scoring, cv=cv,
                                    random_state=cfg.random_seed, n_jobs=cfg.n_jobs)
        search.fit(X, Y[:, i])
        best[target] = search.best_params_
        logger.info("[%s] best CV MSE %.4f with %s", target, -search.best_score_, search.best_params_)
    return best


def params_path(cfg: Config) -> Path:
    return cfg.output_dir / f"best_params_{cfg.model_type}_{cfg.selector}.json"


def save_params(cfg: Config, params: dict[str, dict]) -> Path:
    path = params_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, default=str))
    return path


def load_params(cfg: Config) -> dict[str, dict]:
    """Tuned parameters if `tune` was run for this model/selector, otherwise library defaults."""
    path = params_path(cfg)
    if not path.exists():
        logger.warning("No tuned parameters at %s, using defaults (run the 'tune' command first)", path)
        return {}
    return json.loads(path.read_text())
