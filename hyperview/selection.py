"""Feature selectors. They always live inside a model pipeline, so they are
re-fitted on the training part of every CV split and separately for every target."""
from __future__ import annotations

from functools import partial

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel

from hyperview.config import Config


def _at_most(X, n: int) -> int:
    return min(n, X.shape[1])


def make_selector(cfg: Config, n_jobs: int = 1):
    if cfg.selector == "importance":
        # one forest fit, keep the n most important features
        return SelectFromModel(
            ExtraTreesRegressor(n_estimators=200, random_state=cfg.random_seed, n_jobs=n_jobs),
            max_features=partial(_at_most, n=cfg.n_features), threshold=-float("inf"))
    if cfg.selector == "rfe":
        # recursive elimination, more thorough and much slower
        return RFE(RandomForestRegressor(n_estimators=100, random_state=cfg.random_seed, n_jobs=n_jobs),
                   n_features_to_select=cfg.n_features, step=cfg.rfe_step)
    raise ValueError(f"Unknown selector '{cfg.selector}', use 'importance' or 'rfe'")
