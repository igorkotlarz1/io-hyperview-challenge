"""Loading HYPERVIEW images and labels, writing submissions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hyperview.config import Config


def load_ground_truth(cfg: Config) -> pd.DataFrame:
    """Training labels indexed by sample_index, columns P, K, Mg, pH."""
    path = cfg.ground_truth_csv
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path} (see README, 'Data')")
    return pd.read_csv(path, index_col=0)


def load_wavelengths(cfg: Config) -> pd.DataFrame:
    path = cfg.wavelengths_csv
    if not path.exists():
        raise FileNotFoundError(f"Wavelengths not found: {path}")
    return pd.read_csv(path, index_col=0)


def load_image(path: Path) -> np.ma.MaskedArray:
    """One hyperspectral patch: masked array of shape (bands, height, width).

    The mask is True outside the agricultural field.
    """
    with np.load(path) as npz:
        return np.ma.MaskedArray(**npz)


def list_image_ids(images_dir: Path) -> list[int]:
    """Sorted integer ids of all <id>.npz files in a directory."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir} (see README, 'Data')")
    return sorted(int(p.stem) for p in images_dir.glob("*.npz") if p.stem.isdigit())


def save_submission(predictions: np.ndarray, cfg: Config, name: str) -> Path:
    """Write predictions in the challenge format: sample_index, P, K, Mg, pH."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / f"submission_{name}.csv"
    pd.DataFrame(predictions, columns=list(cfg.targets)).to_csv(path, index_label="sample_index")
    return path
