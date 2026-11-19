"""Tests on a small synthetic dataset in the official HYPERVIEW file layout."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hyperview.__main__ import main
from hyperview.config import Config
from hyperview.evaluation import hyperview_score
from hyperview.features import extract_features, feature_names, masked_gaussian_smoothing

BANDS = 150


def make_cube(rng, size, field_level, background=10_000):
    """A square patch with a circular field; pixels outside the field are masked."""
    yy, xx = np.mgrid[:size, :size]
    field = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size / 2.5) ** 2
    spectrum = 400 + field_level * np.linspace(0.5, 1.5, BANDS)
    data = spectrum[:, None, None] + rng.normal(0, 5, (BANDS, size, size))
    data[:, ~field] = background
    mask = np.broadcast_to(~field, (BANDS, size, size))
    return data.astype(np.int16), mask


def write_dataset(root: Path, n_train=48, n_test=12, seed=0):
    rng = np.random.default_rng(seed)
    train_dir = root / "train_data" / "train_data"
    test_dir = root / "test_data"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    labels = []
    for split_dir, n in ((train_dir, n_train), (test_dir, n_test)):
        for i in range(n):
            level, size = rng.uniform(50, 300), int(rng.integers(10, 24))
            data, mask = make_cube(rng, size, level)
            np.savez_compressed(split_dir / f"{i}.npz", data=data, mask=mask)
            if split_dir == train_dir:  # targets depend on the spectrum level and the field size
                labels.append({"sample_index": i, "P": 20 + 0.3 * level, "K": 100 + 8 * size,
                               "Mg": 150 + 0.1 * level + rng.normal(0, 3), "pH": 6 + level / 300})
    pd.DataFrame(labels).to_csv(root / "train_data" / "train_gt.csv", index=False)
    pd.DataFrame({"band": range(BANDS), "wavelength": np.linspace(462, 942, BANDS)}).to_csv(
        root / "train_data" / "wavelengths.csv", index=False)


def test_masked_smoothing_ignores_background():
    data, mask = make_cube(np.random.default_rng(1), 20, field_level=100, background=10_000)
    valid = ~mask[0]
    smoothed = masked_gaussian_smoothing(data, valid, sigma_xy=2.0)
    assert smoothed[:, valid].max() < 1_000          # no background leaked into the field
    naive = __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter(
        data.astype(float), sigma=(0, 2, 2))
    assert naive[:, valid].max() > 1_000             # plain Gaussian blur does leak


def test_feature_vector_layout():
    cfg = Config()
    data, mask = make_cube(np.random.default_rng(2), 16, field_level=120)
    features = extract_features(np.ma.MaskedArray(data, mask), cfg)
    assert features.shape == (len(feature_names(cfg)),) == (5 * BANDS + 2,)
    assert np.isfinite(features).all()


def test_fully_masked_patch_is_not_dropped():
    data, _ = make_cube(np.random.default_rng(3), 10, field_level=80)
    features = extract_features(np.ma.MaskedArray(data, np.ones_like(data, dtype=bool)), Config())
    assert np.isfinite(features).all()


def test_hyperview_score():
    y = np.array([[1.0, 2, 3, 4], [3, 4, 5, 6]])
    baseline = np.tile(y.mean(axis=0), (2, 1))
    assert hyperview_score(y, baseline, baseline)["score"] == pytest.approx(1.0)
    assert hyperview_score(y, y, baseline)["score"] == pytest.approx(0.0)
    assert hyperview_score(pd.DataFrame(y), y, baseline)["score"] == pytest.approx(0.0)


def test_end_to_end_cli(tmp_path):
    data_dir, out_dir = tmp_path / "data", tmp_path / "outputs"
    write_dataset(data_dir)
    common = ["--data-dir", str(data_dir), "--output-dir", str(out_dir), "--n-features", "15",
              "--tune-iter", "2", "--cv-folds", "3", "--cv-repeats", "1", "--n-jobs", "1"]
    main(["all", *common])

    submission = pd.read_csv(out_dir / "submission_rf.csv")
    assert list(submission.columns) == ["sample_index", "P", "K", "Mg", "pH"]
    assert len(submission) == 12 and submission["sample_index"].tolist() == list(range(12))

    cv = pd.read_csv(out_dir / "cv_rf_importance_mean-d1-d2-d3-std-size.csv")
    assert len(cv) == 3
    assert cv["score"].mean() < 0.5                  # the synthetic targets are learnable
    assert (out_dir / "best_params_rf_importance.json").exists()
    assert (out_dir / "models" / "soil_model_rf.joblib").exists()


def test_knn_model_and_saved_feature_settings(tmp_path):
    from hyperview.features import build_feature_matrix
    from hyperview.models import SoilModel

    data_dir, out_dir = tmp_path / "data", tmp_path / "outputs"
    write_dataset(data_dir, n_train=30, n_test=5)
    main(["submit", "--data-dir", str(data_dir), "--output-dir", str(out_dir), "--model", "knn",
          "--features", "d1", "size", "--n-features", "10", "--n-jobs", "1"])

    # loading with a default config restores the feature groups the model was trained on
    model = SoilModel.load(Config(data_dir=data_dir, output_dir=out_dir), out_dir / "models" / "soil_model_knn.joblib")
    assert model.cfg.feature_groups == ("d1", "size") and model.cfg.model_type == "knn"
    X_test, _ = build_feature_matrix(model.cfg, "test")
    assert model.predict(X_test).shape == (5, 4)
    assert all(len(names) == 10 for names in model.selected_features(feature_names(model.cfg)).values())
