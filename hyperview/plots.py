"""Quick-look plots for a single hyperspectral patch."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_band_with_mask(cube: np.ma.MaskedArray, band: int = 0) -> None:
    """One band with and without the field mask."""
    if not 0 <= band < cube.shape[0]:
        raise ValueError(f"Band must be in [0, {cube.shape[0] - 1}]")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(cube[band].data)
    ax[0].set_title("raw")
    ax[1].imshow(cube[band])
    ax[1].set_title("field mask applied")
    fig.suptitle(f"Band {band}")
    plt.show()


def plot_mean_spectrum(cube: np.ma.MaskedArray, wavelengths: pd.DataFrame) -> None:
    """Mean reflectance of the whole patch vs. the field pixels only."""
    wl = wavelengths.iloc[:, 0].to_numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(wl, cube.data.mean(axis=(1, 2)), "--", label="all pixels (incl. background)")
    plt.plot(wl, cube.mean(axis=(1, 2)), label="field pixels")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Reflectance")
    plt.title("Mean reflectance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_spectrum_derivatives(cube: np.ma.MaskedArray, wavelengths: pd.DataFrame) -> None:
    """Mean field spectrum and its first three finite-difference derivatives."""
    wl = wavelengths.iloc[:, 0].to_numpy()
    spectrum = np.ma.getdata(cube.mean(axis=(1, 2)))
    d1 = np.gradient(spectrum)
    d2 = np.gradient(d1)
    d3 = np.gradient(d2)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(wl, spectrum)
    ax[0].set_title("Mean field spectrum")
    for values, label in ((d1, "1st derivative"), (d2, "2nd derivative"), (d3, "3rd derivative")):
        ax[1].plot(wl, values, label=label)
    ax[1].set_title("Derivatives")
    ax[1].legend()
    fig.supxlabel("Wavelength [nm]")
    plt.show()


def plot_band_correlation(cube: np.ma.MaskedArray) -> None:
    """Pearson correlation between bands across the field pixels."""
    bands = cube.shape[0]
    pixels = cube.reshape(bands, -1)
    valid = ~np.ma.getmaskarray(pixels).any(axis=0)
    corr = np.corrcoef(np.ma.getdata(pixels)[:, valid])
    plt.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(label="correlation")
    plt.xlabel("band")
    plt.ylabel("band")
    plt.title("Band-to-band correlation")
    plt.show()
