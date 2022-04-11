import os
from abc import abstractmethod, ABC

import numpy as np
import scipy
from tqdm import tqdm

from utils.dataset import get_split, get_final_split, get_cv_splits


class BaseModel(ABC):

    @abstractmethod
    def init_model(self) -> None:
        pass

    @abstractmethod
    def fit(self, x, y, val_x=None, val_y=None) -> None:
        pass

    @abstractmethod
    def predict(self, y) -> np.array:
        pass

    @staticmethod
    def get_score(y, y_true) -> float:
        return scipy.stats.spearmanr(y, y_true)[0]

    def fit_predict(self) -> (np.ndarray, np.ndarray):
        print(f'Train and test {self.__class__.__name__} model')
        self.init_model()

        (x_train, y_train), (x_test, y_test) = get_split()

        self.fit(x_train, y_train, x_test, y_test)

        y_pred = self.predict(x_test)
        print('test spearman correlation score:', self.get_score(y_pred, y_test))
        print()
        return y_pred, y_test

    def cross_validate(self) -> np.ndarray:
        print(f'Cross-validate {self.__class__.__name__} model')
        val_scores = []
        for (x_train, y_train), (x_val, y_val) in tqdm(get_cv_splits()):
            self.init_model()
            self.fit(x_train, y_train, x_val, y_val)
            val_scores.append(self.get_score(self.predict(x_val), y_val))
        print(f'Mean val score {sum(val_scores)/len(val_scores):.3f}, scores: {val_scores}')
        print()
        return np.array(val_scores)

    def create_submission(self):
        self.init_model()

        (x_train, y_train), (x_test, test_df) = get_final_split()

        self.fit(x_train, y_train)

        y_pred = self.predict(x_test)

        save_dir = '../data/submissions'
        file_name = 'gex_predicted.csv'
        zip_name = "Kasak_Liine_Project1.zip"
        save_path = f'{save_dir}/{zip_name}'
        compression_options = dict(method="zip", archive_name=file_name)

        test_df['gex_predicted'] = y_pred.flatten().tolist()
        print(f'Saving submission to path {os.path.abspath(save_dir)}')
        test_df[['gene_name', 'gex_predicted']].to_csv(save_path, compression=compression_options)
