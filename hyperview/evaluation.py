"""HYPERVIEW score and repeated K-fold evaluation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from hyperview.config import Config
from hyperview.models import MeanBaseline, SoilModel

logger = logging.getLogger(__name__)


def hyperview_score(y_true, y_pred, y_baseline, targets=("P", "K", "Mg", "pH")) -> pd.Series:
    """Per-target MSE divided by the baseline MSE, and their mean (lower is better, 1.0 = baseline)."""
    y_true, y_pred, y_baseline = (np.asarray(a, dtype=float) for a in (y_true, y_pred, y_baseline))
    ratios = ((y_true - y_pred) ** 2).mean(axis=0) / ((y_true - y_baseline) ** 2).mean(axis=0)
    scores = pd.Series(ratios, index=list(targets))
    scores["score"] = ratios.mean()
    return scores


def cross_validate(cfg: Config, X: np.ndarray, Y: np.ndarray, params: dict[str, dict] | None = None) -> pd.DataFrame:
    """Repeated K-fold estimate of the HYPERVIEW score; one row per fold.

    Everything that learns from data (feature selection, model, baseline mean)
    is fitted on the training folds only.
    """
    splitter = RepeatedKFold(n_splits=cfg.cv_folds, n_repeats=cfg.cv_repeats, random_state=cfg.random_seed)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = SoilModel(cfg, params).fit(X[train_idx], Y[train_idx])
        baseline = MeanBaseline().fit(X[train_idx], Y[train_idx])
        scores = hyperview_score(Y[test_idx], model.predict(X[test_idx]), baseline.predict(X[test_idx]), cfg.targets)
        rows.append(scores)
        logger.info("fold %2d/%d  score %.4f", fold, cfg.cv_folds * cfg.cv_repeats, scores["score"])
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean ± standard error over folds."""
    return pd.DataFrame({"mean": results.mean(), "std_err": results.std(ddof=1) / np.sqrt(len(results))})
