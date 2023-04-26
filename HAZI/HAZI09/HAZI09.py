# imports
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix as conf_matrix
from sklearn.datasets import load_digits as load_d


class KMeansOnDigits:
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state


    # Készíts egy függvényt ami betölti a digits datasetet
    # NOTE: használd az sklearn load_digits-et
    # Függvény neve: load_digits()
    # Függvény visszatérési értéke: a load_digits visszatérési értéke

    def load_digits(self):
        self.digits = load_d()


    #digits = pd.DataFrame(data=ds.data, columns=ds.feature_names)
    #digits['target'] = ds['target']

    # Készíts egy függvényt ami létrehoz egy KMeans model-t 10 db cluster-el
    # NOTE: használd az sklearn Kmeans model-jét (random_state legyen 0)
    # Miután megvan a model predict-elj vele
    # NOTE: használd a fit_predict-et
    # Függvény neve: predict(n_clusters:int,random_state:int,digits)
    # Függvény visszatérési értéke: (model:sklearn.cluster.KMeans,clusters:np.ndarray)

    def predict(self):
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = model.fit_predict(self.digits.data)

    # Készíts egy függvényt ami visszaadja a predictált cluster osztályokat
    # NOTE: amit a predict-ből visszakaptunk "clusters" azok lesznek a predictált cluster osztályok
    # HELP: amit a model predictált cluster osztályok még nem a labelek, hanem csak random cluster osztályok,
    #       Hogy label legyen belőlük:
    #       1. készíts egy result array-t ami ugyan annyi elemű mint a predictált cluster array
    #       2. menj végig mindegyik cluster osztályon (0,1....9)
    #       3. készíts egy maszkot ami az adott cluster osztályba tartozó elemeket adja vissza
    #       4. a digits.target-jét indexeld meg ezzel a maszkkal
    #       5. számold ki ennel a subarray-nek a móduszát
    #       6. a result array-ben tedd egyenlővé a módusszal azokat az indexeket ahol a maszk True
    #       Erre azért van szükség mert semmi nem biztosítja nekünk azt, hogy a "0" cluster a "0" label lesz, lehet, hogy az "5" label lenne az.

    # Függvény neve: get_labels(clusters:np.ndarray, digits)
    # Függvény visszatérési értéke: labels:np.ndarray

    def get_labels(self):
        result = []
        for i in self.clusters:
            mask = (self.clusters == i)
            mod = mode(self.digits.target[mask])[0][0]
            result[mask] = mod
        self.labels = result

    # Készíts egy függvényt ami kiszámolja a model accuracy-jét
    # Függvény neve: calc_accuracy(target_labels:np.ndarray,predicted_labels:np.ndarray)
    # Függvény visszatérési értéke: accuracy:float
    # NOTE: Kerekítsd 2 tizedes jegyre az accuracy-t

    def calc_accuracy(self, target_labels: np.ndarray, predicted_labels: np.ndarray):
        self.accuracy = np.round(accuracy_score(target_labels, predicted_labels), decimals=2)

    # Készíts egy confusion mátrixot és plot-old seaborn segítségével
    def confusion_matrix(self, target_labels: np.ndarray, predicted_labels: np.ndarray):
        self.mat = conf_matrix(target_labels, predicted_labels)
