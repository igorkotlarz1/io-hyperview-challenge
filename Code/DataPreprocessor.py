import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from config import Config
from DataManager import DataManager

class Preprocessor:
    def __init__(self, config: Config):
        self.cfg = config
        #self.dm = DataManager(config)

    def _apply_gauss_smoothing(self, spectral_arr: np.ndarray): 
        return gaussian_filter(spectral_arr, sigma=self.cfg.SIGMA_PARAM)

    def _calculate_mean_spectrum(self, spectral_arr: np.ma.MaskedArray):
        mean_spectrum = spectral_arr.mean(axis=(1,2))

        if np.ma.is_masked(mean_spectrum):
            return None
        return mean_spectrum.data

    def _apply_sav_gol_filter(self, mean_spectrum: np.ndarray):
        sg_window = self.cfg.SG_WINDOW
        sg_poly = self.cfg.SG_POLY
        sg_deriv = self.cfg.SG_DERIV

        return savgol_filter(mean_spectrum, window_length=sg_window, polyorder=sg_poly, deriv=sg_deriv)

    def process_image(self, spectral_arr: np.ma.MaskedArray):
        processed_data = spectral_arr.copy()

        blurred_data = self._apply_gauss_smoothing(spectral_arr.data)
        processed_data.data[:] = blurred_data
        mean_spectrum = self._calculate_mean_spectrum(processed_data)

        if mean_spectrum is None:
            return None
        
        derivative = self._apply_sav_gol_filter(mean_spectrum)
        return derivative

    def process_dataset(self, data_manager: DataManager, gt_df: pd.DataFrame):
        X_features = []
        y_labels = []
        valid_indices = []

        target_cols = list(self.cfg.TARGET_COLS)

        for id, row in gt_df.iterrows():
            spectral_arr = data_manager.load_image(id)          

            if spectral_arr is None:
                continue

            processed_arr = self.process_image(spectral_arr)

            if processed_arr is not None:
                X_features.append(processed_arr)
                y_labels.append(row[target_cols].values)
                valid_indices.append(id)

        X = np.array(X_features)
        y = np.array(y_labels)

        return X, y, valid_indices




    