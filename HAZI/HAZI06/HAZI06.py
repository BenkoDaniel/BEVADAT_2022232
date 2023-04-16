import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from NJCleaner import NJCleaner
from GYAK.GYAK06.GYAK06 import DecisionTreeClassifier

default_csv_path = 'data/2018_03.csv'
processed_csv_path = 'data/NJ.csv'

cleaner = NJCleaner(default_csv_path)
cleaner.prep_df(processed_csv_path)


data = pd.read_csv(processed_csv_path)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=41)

dtclassifier = DecisionTreeClassifier(min_samples_split=1, max_depth=4)
dtclassifier.fit(x_train, y_train)

y_pred = dtclassifier.predict(x_test)

print(accuracy_score(y_test, y_pred))

