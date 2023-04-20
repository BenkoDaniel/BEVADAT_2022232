import numpy as np
import pandas as pd
from LinearRegressionSkeleton import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

X = df['petal width (cm)'].values
y = df['sepal length (cm)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = reg.evaluate(X_test, y_pred)

plt.scatter(X_test, y_test)
plt.plot([min(X_test), max(X_test)], [min(y_pred), max(y_pred)], color='red') # predicted
plt.show()

print(f'Mean Squared Error: {mse}')

