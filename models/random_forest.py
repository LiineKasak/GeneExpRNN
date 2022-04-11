from abc import ABC

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseModel


class RandomForest(BaseModel, ABC):

    def __init__(self, max_depth: int = 10, n_estimators: int = 20):
        self.model = None
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        self.init_model()

    def init_model(self) -> None:
        self.model = RandomForestRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, bootstrap=True,
                                           n_jobs=-1, random_state=42)

    def fit(self, x, y, x_val=None, y_val=None) -> None:
        self.model.fit(self._reshape(x), y)

    def predict(self, x) -> np.ndarray:
        return self.model.predict(self._reshape(x))

    @staticmethod
    def _reshape(x: np.ndarray) -> np.ndarray:
        n_genes, n_features, n_bins = x.shape
        return x.reshape(n_genes, n_features * n_bins)


if __name__ == '__main__':
    model = RandomForest()
    model.fit_predict()
    model.cross_validate()
