import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from config import Config

class ModelManager:
    def __init__(self, config: Config):
        self.cfg = config
        self.models = {}
        self.best_params = {}
        
        self.param_grid = {
            'n_estimators': [200, 300, 400, 500, 700, 1000],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features' : ['log2','sqrt', None]
        }
        
        
    def tune_fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose=False):
        target_cols = list(self.cfg.TARGET_COLS)
        SEED = self.cfg.RANDOM_SEED

        for i, target_param in enumerate(target_cols):
            #subsetting y_train, y_test to obtain single soil param each iteration
            y_target = y_train[:,i]

            base_model = RandomForestRegressor(random_state=SEED, n_jobs=-1)
            search = RandomizedSearchCV(estimator=base_model, param_distributions=self.param_grid, n_iter = 100,random_state=SEED,
                                        cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)           
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
    
    def get_baseline_predictions(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
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