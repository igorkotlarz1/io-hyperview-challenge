import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from config import Config

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

class ModelManager:
    def __init__(self, config: Config, model_type: str = "rf"):
        self.cfg = config
        self.models = {}
        self.best_params = {}
        self.model_type = model_type
        
        self.param_grid = {
            'n_estimators': [200, 300, 400, 500, 700, 1000],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features' : ['log2','sqrt', None]
        }
    
    def _get_model_conf(self):
        SEED = self.cfg.RANDOM_SEED

        if self.model_type == "rf":
            params = {
                'n_estimators': [200, 300, 400, 500, 700, 1000],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features' : ['log2','sqrt', None]
            }
            model = RandomForestRegressor(random_state=SEED, n_jobs=-1)
            return model, params
        
        elif self.model_type == "xgb":
            if XGBRegressor is None:
                raise ValueError('XGB library is not installed!')
            params = {
                'n_estimators': [100, 300, 500, 1000],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 1, 10],
                'reg_lambda': [1, 10]
            }
            model = XGBRegressor(seed=SEED)
            return model, params
        
        else:
            raise ValueError('Unknown model type')
        
    def tune_fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose=False):
        target_cols = list(self.cfg.TARGET_COLS)
        SEED = self.cfg.RANDOM_SEED

        _, params = self._get_model_conf()

        for i, target_param in enumerate(target_cols):
            #subsetting y_train, y_test to obtain single soil param each iteration
            y_target = y_train[:,i]
            base_model,  _ = self._get_model_conf()
            #base_model = RandomForestRegressor(random_state=SEED, n_jobs=-1)
            search = RandomizedSearchCV(estimator=base_model, param_distributions=params, n_iter = 20, random_state=SEED, 
                                        cv=5, scoring=self.cfg.CV_SCORING_MAE, n_jobs=-1, verbose=3)            #self.param_grid
            search.fit(X_train, y_target)

            best_model = search.best_estimator_
            self.models[target_param] = best_model

            if verbose:
                print(f'Best params for [{target_param}] parameter: {search.best_params_}\n')


    def predict(self, X_test: np.ndarray) -> pd.DataFrame:
        target_cols = list(self.cfg.TARGET_COLS)
        SEED = self.cfg.RANDOM_SEED

        if not self.models:
            raise RuntimeError('You need to fit models first!')
        
        predictions = {}

        for target_param in target_cols:
            model = self.models[target_param]
            y_pred = model.predict(X_test)

            predictions[target_param] = y_pred

        return pd.DataFrame(predictions)
    
    @staticmethod
    def get_baseline_predictions(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
        baseline_reg = BaselineRegressor()
        baseline_reg.fit(X_train, y_train)

        return baseline_reg.predict(X_test)
    
    def save_models(self, filename:str = 'models.pkl'):
        path = os.path.join(self.cfg.SUBMISSION_DIR,'Models',filename)

        joblib.dump(self.models, path)
        print(f'Successfully saved models to: {path}!')

    def load_models(self, filename:str = 'models.pkl'):
        path = os.path.join(self.cfg.SUBMISSION_DIR,'Models',filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Failed to find model file: {path}!')
        
        self.models = joblib.load(path)

class BaselineRegressor:
    """
    Baseline regressor, which calculates the mean value of the target from the training
    data and returns it for each testing sample.
    """
    def __init__(self):
        self.mean = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.mean = np.mean(y_train, axis=0)
        self.classes_count = y_train.shape[1]
        return self

    def predict(self, X_test: np.ndarray):
        return np.full((len(X_test), self.classes_count), self.mean)