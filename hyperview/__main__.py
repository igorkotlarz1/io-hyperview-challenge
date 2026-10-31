"""Command-line interface.

    python -m hyperview features            # extract + cache features for train and test
    python -m hyperview tune                # tune hyperparameters per target (MSE)
    python -m hyperview cv                  # repeated K-fold HYPERVIEW score
    python -m hyperview submit              # fit on all training data, write the submission
    python -m hyperview all                 # all of the above

Common options: --model rf|xgb, --selector importance|rfe, --n-features N,
--features mean d1 d2 d3 std size, --data-dir PATH, --output-dir PATH.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hyperview.config import FEATURE_GROUPS, Config
from hyperview.data import load_ground_truth, save_submission
from hyperview.evaluation import cross_validate, summarize
from hyperview.features import build_feature_matrix
from hyperview.models import SoilModel, load_params, save_params, tune_hyperparameters

logger = logging.getLogger("hyperview")


def load_training_set(cfg: Config):
    X, ids = build_feature_matrix(cfg, "train")
    Y = load_ground_truth(cfg).loc[ids, list(cfg.targets)].to_numpy(dtype=float)
    return X, Y


def run_features(cfg: Config) -> None:
    for split in ("train", "test"):
        X, _ = build_feature_matrix(cfg, split)
        logger.info("%s features: %s", split, X.shape)


def run_tune(cfg: Config) -> None:
    X, Y = load_training_set(cfg)
    path = save_params(cfg, tune_hyperparameters(cfg, X, Y))
    logger.info("Saved tuned parameters to %s", path)


def run_cv(cfg: Config) -> None:
    X, Y = load_training_set(cfg)
    results = cross_validate(cfg, X, Y, load_params(cfg))
    name = f"cv_{cfg.model_type}_{cfg.selector}_{'-'.join(cfg.feature_groups)}"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg.output_dir / f"{name}.csv", index_label="fold")
    summary = summarize(results)
    print(f"\nHYPERVIEW score, {cfg.cv_repeats}×{cfg.cv_folds}-fold CV ({name}):")
    print(summary.round(4).to_string())


def run_submit(cfg: Config) -> None:
    X, Y = load_training_set(cfg)
    X_test, test_ids = build_feature_matrix(cfg, "test")
    model = SoilModel(cfg, load_params(cfg)).fit(X, Y)
    model.save(cfg.output_dir / "models" / f"soil_model_{cfg.model_type}.joblib")
    path = save_submission(model.predict(X_test), cfg, cfg.model_type)
    logger.info("Wrote %s (%d test samples, ids %d..%d)", path, len(test_ids), test_ids[0], test_ids[-1])


COMMANDS = {"features": [run_features], "tune": [run_tune], "cv": [run_cv], "submit": [run_submit],
            "all": [run_features, run_tune, run_cv, run_submit]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m hyperview", description="HYPERVIEW soil parameter estimation")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--data-dir", type=Path, default=Config.data_dir)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--model", choices=["rf", "xgb"], default=Config.model_type)
    parser.add_argument("--selector", choices=["importance", "rfe"], default=Config.selector)
    parser.add_argument("--n-features", type=int, default=Config.n_features)
    parser.add_argument("--features", nargs="+", choices=FEATURE_GROUPS, default=list(FEATURE_GROUPS),
                        help="feature groups to use (default: all)")
    parser.add_argument("--tune-iter", type=int, default=Config.tune_iter)
    parser.add_argument("--cv-folds", type=int, default=Config.cv_folds)
    parser.add_argument("--cv-repeats", type=int, default=Config.cv_repeats)
    parser.add_argument("--seed", type=int, default=Config.random_seed)
    parser.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = Config(data_dir=args.data_dir, output_dir=args.output_dir, model_type=args.model,
                 selector=args.selector, n_features=args.n_features, feature_groups=tuple(args.features),
                 tune_iter=args.tune_iter, cv_folds=args.cv_folds, cv_repeats=args.cv_repeats,
                 random_seed=args.seed, n_jobs=args.n_jobs)
    for step in COMMANDS[args.command]:
        step(cfg)


if __name__ == "__main__":
    main()
