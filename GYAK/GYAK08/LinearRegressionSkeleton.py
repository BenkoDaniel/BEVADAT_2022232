import numpy as np



class LinearRegression:
    epochs: int
    lr: float
    m: float
    c: float

    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr
        self.m = 0
        self.c = 0

    def fit(self, X: np.array, y: np.array):
        n = float(len(X))  # Number of elements in X

        # Performing Gradient Descent
        losses = []
        for i in range(self.epochs):
            y_pred = self.m * X + self.c  # The current predicted value of Y

            residuals = y - y_pred
            loss = np.sum(residuals ** 2)
            losses.append(loss)
            D_m = (-2 / n) * sum(X * residuals)  # Derivative wrt m
            D_c = (-2 / n) * sum(residuals)  # Derivative wrt c
            self.m = self.m - self.lr * D_m  # Update m
            self.c = self.c - self.lr * D_c  # Update c



    def predict(self, X):
        preds = self.m * X + self.c
        return preds

    def evaluate(self, X, y):
        mse = np.mean((self.predict(X) - y) ** 2)
        return mse

