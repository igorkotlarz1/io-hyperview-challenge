"""Turn a hyperspectral patch (150 bands × H × W, masked) into a feature vector.

Pipeline per image:
1. Spatial Gaussian smoothing that respects the field mask (normalized
   convolution: pixels outside the field never leak into the field).
2. Mean spectrum over the field pixels.
3. Savitzky-Golay first derivative of the mean spectrum, then 2nd and 3rd
   derivatives by finite differences.
4. Per-band spatial standard deviation (field heterogeneity).
5. Field size (number of valid pixels).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from hyperview.config import FEATURE_GROUPS, Config
from hyperview.data import list_image_ids, load_image

logger = logging.getLogger(__name__)


def field_pixel_mask(cube: np.ma.MaskedArray) -> np.ndarray:
    """(H, W) boolean array, True for pixels inside the field in every band."""
    mask = np.ma.getmaskarray(cube)
    valid = ~mask.any(axis=0)
    if not valid.any():  # fully masked patch: fall back to all pixels instead of dropping it
        valid = np.ones(cube.shape[1:], dtype=bool)
    return valid


def masked_gaussian_smoothing(data: np.ndarray, valid: np.ndarray, sigma_xy: float) -> np.ndarray:
    """Spatial Gaussian blur of every band using only pixels where ``valid`` is True.

    Plain ``gaussian_filter`` on the raw array mixes background pixels into the
    field edges. Normalized convolution divides the blurred signal by the blurred
    weight map, so each output pixel is a weighted mean of valid neighbours only.
    """
    if sigma_xy <= 0:
        return data.astype(np.float64)
    weights = valid.astype(np.float64)
    sigma = (0, sigma_xy, sigma_xy)
    numerator = gaussian_filter(data.astype(np.float64) * weights[None], sigma=sigma)
    denominator = gaussian_filter(np.broadcast_to(weights, data.shape).astype(np.float64), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = numerator / denominator
    return np.where(denominator > 1e-12, smoothed, 0.0)


def extract_features(cube: np.ma.MaskedArray, cfg: Config) -> np.ndarray:
    """Feature vector for one patch, in the order of ``feature_names``."""
    valid = field_pixel_mask(cube)
    smoothed = masked_gaussian_smoothing(np.ma.getdata(cube), valid, cfg.gauss_sigma_xy)
    field = smoothed[:, valid]                      # (bands, n_field_pixels)

    mean_spectrum = field.mean(axis=1)
    d1 = savgol_filter(mean_spectrum, window_length=cfg.sg_window, polyorder=cfg.sg_poly, deriv=1)
    d2 = np.gradient(d1)
    d3 = np.gradient(d2)
    n_pixels = float(valid.sum())

    groups = {
        "mean": mean_spectrum,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "std": field.std(axis=1),
        "size": np.array([n_pixels, np.log1p(n_pixels)]),
    }
    return np.concatenate([groups[g] for g in cfg.feature_groups])


def feature_names(cfg: Config, n_bands: int = 150) -> list[str]:
    names = []
    for group in cfg.feature_groups:
        if group == "size":
            names += ["field_pixels", "log_field_pixels"]
        else:
            names += [f"{group}_b{b:03d}" for b in range(n_bands)]
    return names


def build_feature_matrix(cfg: Config, split: str, use_cache: bool = True) -> tuple[np.ndarray, list[int]]:
    """Features for every image of a split ("train" or "test"), cached in outputs/features/."""
    unknown = set(cfg.feature_groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}. Choose from {FEATURE_GROUPS}")

    cache = cfg.output_dir / "features" / f"{split}_{cfg.feature_cache_key()}.npz"
    if use_cache and cache.exists():
        with np.load(cache) as npz:
            logger.info("Loaded cached features: %s", cache)
            return npz["X"], npz["ids"].tolist()

    images_dir = cfg.train_images_dir if split == "train" else cfg.test_images_dir
    ids = list_image_ids(images_dir)
    rows = []
    for n, image_id in enumerate(ids, start=1):
        rows.append(extract_features(load_image(Path(images_dir) / f"{image_id}.npz"), cfg))
        if n % 200 == 0 or n == len(ids):
            logger.info("%s: %d/%d images", split, n, len(ids))
    X = np.vstack(rows)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, ids=np.array(ids))
    return X, ids
