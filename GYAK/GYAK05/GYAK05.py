%pip install scipy
%pip install scikit_learn
%pip install seaborn

import numpy as np
from typing import Tuple
from scipy.stats import mode

from sklearn.metrics import confusion_matrix
import seaborn as sns

#property elnevezes: k_neighbors

class KNNClasifier:

    def __init__(self, k:int, test_split_raio:float) -> None:
        self.k = k
        self.test_split_ratio = test_split_raio


    def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)

        dataset = np.genfromtxt(path, delimiter=',')
        np.random.shuffle(dataset)
        x, y = dataset[:, :-1], dataset[:, -1]
        return x, y

    x, y = load_csv('iris.csv')
    np.nanmean(x, axis=0), np.nanvar(x, axis=0)

    x[np.isnan(x)] = 3.5

    y = np.delete(y, np.where(x < 0.0)[0], axis=0)
    y = np.delete(y, np.where(x > 10.0)[0], axis=0)
    x = np.delete(x, np.where(x < 0.0)[0], axis=0)
    x = np.delete(x, np.where(x < 0.0)[0], axis=0)

    def train_test_split(features: np.ndarray, labels: np.ndarray, text_split_ratio: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        test_size = int(len(features)) * text_split_ratio
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        x_train, y_train = features[:train_size, :], labels[:train_size]
        x_test, y_test = features[train_size:, :], labels[train_size:]
        return x_train, y_train, x_test, y_test

    def predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int) -> np.ndarray:
        for x_test_element in x_test:
            distances = euclidean(x_train, x_test_element)
            distances = np.array(sorted(zip(distances, y_train)))

            label_pred = mode(distances[:k, 1], keepdims=False)
            labels_pred.append(label_pred)
        return np.addray(labels_pred, dtype=np.int64)

    def euclidean(points: np.ndarray, element_of_x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((points - element_of_x) ** 2, axis=1))


    def accuracy(y_test: np.ndarray, y_preds: np.ndarray) -> float:
        true_positive = (y_test == y_preds).sum()
        return true_positive

    def confusion_matrix(y_test: np.ndarray, y_preds: np.ndarray):
        conf_matrix = confusion_matrix(y_test, y_preds)
        sns.heatmap(conf_matrix, annot=True)
        return conf_matrix





