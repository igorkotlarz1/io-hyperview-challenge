import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from config import Config


class KNNPhManager:
    def __init__(self, config: Config):
        self.cfg = config


        # for knn need for preskaling

    #first need to scale then choose our model knn
        self.pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor()) ])

        # finding opitmals here is list of things to test and from them choose the most optimal
        self.param_grid = {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 20], #how many neighbours show we take to account
            'knn__weights': ['uniform', 'distance'], #should every value be the same or should have subjects that are further away have less value
            'knn__metric': ['euclidean']
        }

    def fit_tune(self, X_train: np.ndarray, y_train: np.ndarray):

        #training only on pH



        target_idx = self.cfg.TARGET_COLS.index('pH') #getting column where is ph (number)
       #only pH getting
        y_ph = y_train[:, target_idx]




        grid = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=self.cfg.CV_FOLDS,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_ph)
        self.model = grid.best_estimator_


        print(f"Best  KNN parameters for pH: {grid.best_params_}")
        print(f"Best score (MSE) : {-grid.best_score_:.4f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
              return self.model.predict(X_test)