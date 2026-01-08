import pandas as pd
import numpy as np
import typing
import os
from config import Config

class DataManager:

    def __init__(self, config: Config):
        self.cfg = config
        self._ensure_submission_dir()

    def _ensure_submission_dir(self):
        os.makedirs(self.cfg.SUBMISSION_DIR, exist_ok=True)

    def load_ground_truth(self) -> pd.DataFrame: 
        gt_path = self.cfg.GT_PATH

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f'Could not find GT file: {gt_path}')
        
        return pd.read_csv(gt_path)
    
    def load_wavelengths(self) -> pd.DataFrame:
        wl_path = self.cfg.WL_PATH

        if not os.path.exists(wl_path):
            raise FileNotFoundError(f'Could not find GT file: {wl_path}')
        
        return pd.read_csv(wl_path)
    
    def load_image(self, image_id:int) -> np.ma.MaskedArray:
        img_path = os.path.join(self.cfg.TRAIN_DATA_PATH,f'{image_id}.npz')

        if not os.path.exists(img_path):
            print(f'Could not find file: {img_path}')
            return None
        
        try: 
            with np.load(img_path) as npz:
                spectral_arr = np.ma.MaskedArray(**npz)
                return spectral_arr
        except Exception as e:
            print(f'Error while loading: {img_path}')
            return None
 

    def save_submission(self, predictions: pd.DataFrame, filename: str = "submission.csv"):       
        if not filename.endswith('.csv'):
            filename += '.csv'
        path = os.path.join(self.cfg.SUBMISSION_DIR, filename)

        predictions.to_csv(path)
        print(f'Succesfully saved submission file to: {path}')

