from abc import ABC

import numpy as np
from sklearn import svm

from models.base_model import BaseModel


class SVM(BaseModel, ABC):

    def __init__(self):
        self.model = None

        self.init_model()

    def init_model(self) -> None:
        self.model = svm.SVR(kernel='rbf', C=10)

    def fit(self, x, y, x_val=None, y_val=None) -> None:
        self.model.fit(self._reshape(x), y)

    def predict(self, x) -> np.ndarray:
        return self.model.predict(self._reshape(x))

    @staticmethod
    def _reshape(x: np.ndarray) -> np.ndarray:
        n_genes, n_features, n_bins = x.shape
        return x.reshape(n_genes, n_features * n_bins)


if __name__ == '__main__':
    model = SVM()
    model.cross_validate()
