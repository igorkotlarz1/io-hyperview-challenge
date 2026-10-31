"""All experiment settings in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TARGETS = ("P", "K", "Mg", "pH")
FEATURE_GROUPS = ("mean", "d1", "d2", "d3", "std", "size")


@dataclass
class Config:
    # Paths (layout of the official HYPERVIEW download, see README)
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    random_seed: int = 123

    # Preprocessing
    gauss_sigma_xy: float = 2.0     # spatial smoothing only, bands are never mixed
    sg_window: int = 11             # Savitzky-Golay window
    sg_poly: int = 2                # Savitzky-Golay polynomial order
    feature_groups: tuple[str, ...] = FEATURE_GROUPS

    # Feature selection (fitted inside each model pipeline, per target)
    selector: str = "importance"    # "importance" (fast) or "rfe" (slow)
    n_features: int = 50
    rfe_step: float = 0.1           # fraction of features removed per RFE iteration

    # Models
    model_type: str = "rf"          # "rf", "xgb" or "knn"
    scoring: str = "neg_mean_squared_error"  # HYPERVIEW score is MSE-based
    tune_iter: int = 30
    tune_folds: int = 5
    cv_folds: int = 5
    cv_repeats: int = 3
    n_jobs: int = -1

    targets: tuple[str, ...] = field(default=TARGETS)

    @property
    def train_images_dir(self) -> Path:
        return self.data_dir / "train_data" / "train_data"

    @property
    def test_images_dir(self) -> Path:
        return self.data_dir / "test_data"

    @property
    def ground_truth_csv(self) -> Path:
        return self.data_dir / "train_data" / "train_gt.csv"

    @property
    def wavelengths_csv(self) -> Path:
        return self.data_dir / "train_data" / "wavelengths.csv"

    def feature_cache_key(self) -> str:
        groups = "-".join(self.feature_groups)
        return f"s{self.gauss_sigma_xy:g}_w{self.sg_window}_p{self.sg_poly}_{groups}"
