import numpy as np
from typing import Tuple
from scipy.stats import mode

from sklearn.metrics import confusion_matrix
#import seaborn as sns

#property elnevezes: k_neighbors

class KNNClassifier:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_preds: np.ndarray
    k: int
    @property
    def k_neighbors(self):
        return self.k

    test_split_ratio: float


    def __init__(self, k:int, test_split_raio:float) -> None:
        self.k = k
        self.test_split_ratio = test_split_raio


    @staticmethod
    def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)

        dataset = np.genfromtxt(path, delimiter=',')
        np.random.shuffle(dataset)
        x, y = dataset[:, :-1], dataset[:, -1]
        return x, y

    def train_set_split(self, features: np.ndarray, labels: np.ndarray):
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train, self.y_train = features[:train_size, :], labels[:train_size]
        self.x_test, self.y_test = features[train_size:, :], labels[train_size:]

    def euclidean(self, element_of_x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((self.x_train - element_of_x) ** 2, axis=1))

    def predict(self, x_test: np.ndarray):
        preds = []
        for x_test_element in x_test:
            distances = self.euclidean(x_test_element)
            distances = np.array(sorted(zip(distances, self.y_train)))

            label_pred = mode(distances[:self.k, 1], keepdims=False)
            preds.append(label_pred)
            self.y_preds = np.array(preds, dtype=np.int32)

    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive

    def confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        #sns.heatmap(conf_matrix, annot=True)
        return np.ndarray(conf_matrix)






