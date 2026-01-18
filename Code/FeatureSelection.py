from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, VarianceThreshold
import numpy as np

from config import Config 

class FeatureSelector:  
    def __init__(self, config: Config,  n_features: int = 50, step: int = 5):
        self.cfg = config
        self.n_features = n_features
        self.step = step
        self.support_ = None
        self.selector_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        seed = self.cfg.RANDOM_SEED

        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=seed)
        self.selector_ = RFE(estimator=rf, n_features_to_select=self.n_features, step=self.step, verbose=1)

        self.selector_.fit(X, y)
        self.support_ = self.selector_.support_

        return self

    def transform(self, X: np.ndarray):
        if self.support_ is None:
            raise RuntimeError('You need to fit the selector first!')
        return X[:, self.support_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        return self.transform(X)

    def get_indices(self):
        if self.support_ is None:
            raise RuntimeError('You need to fit the selector first!')
        selected_indices = np.where(self.selector_.get_support())[0]
        return np.sort(selected_indices)
    
    def get_feature_ranking(self):
        if self.selector_ is None:
            raise RuntimeError('You need to fit the selector first!')
        return self.selector_.ranking_
