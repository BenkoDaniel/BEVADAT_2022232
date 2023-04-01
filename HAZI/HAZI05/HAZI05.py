import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import confusion_matrix


class KNNClassifier:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    y_preds: pd.DataFrame
    k: int

    @property
    def k_neighbors(self):
        return self.k

    test_split_ratio: float

    def __init__(self, k: int, test_split_raio: float) -> None:
        self.k = k
        self.test_split_ratio = test_split_raio

    @staticmethod
    def load_csv(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = pd.read_csv(path, delimiter=',')
        shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x, y = shuffled[:, :-1], shuffled[:, -1]
        return x, y

    def train_set_split(self, features: pd.DataFrame, labels: pd.DataFrame):
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train, self.y_train = features[:train_size, :], labels[:train_size]
        self.x_test, self.y_test = features[train_size:, :], labels[train_size:]

    def euclidean(self, element_of_x: pd.DataFrame) -> pd.DataFrame:
        return pd.sum((self.x_train - element_of_x) ** 2, axis=1)**0.5

    def predict(self, x_test: pd.DataFrame):
        preds = []
        for x_test_element in x_test:
            distances = self.euclidean(x_test_element)
            distances = pd.array(sorted(zip(distances, self.y_train)))

            label_pred = distances[:self.k, 1].mode(keepdims=False)
            preds.append(label_pred)
            self.y_preds = pd.DataFrame(preds)

    def accuracy(self) -> float:
        true_positive = (self.y_test.reset_index(drop=True) == self.y_preds).sum()
        return true_positive

    def plot_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        return np.ndarray(conf_matrix)

    def best_k(self) -> Tuple:
        ac_list = []
        for i in range(1, 21):
            KNNClassifier(i, self.test_split_ratio)
            ac_list.append(tuple((i, round(KNNClassifier.accuracy(self), 2))))
        return max(ac_list)



