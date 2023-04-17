import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from NJCleaner import NJCleaner
from GYAK.GYAK06.GYAK06 import DecisionTreeClassifier


default_csv_path = 'data/2018_03.csv'
processed_csv_path = 'data/NJ.csv'

'''
cleaner = NJCleaner(default_csv_path)
cleaner.prep_df(processed_csv_path)
'''

data = pd.read_csv(processed_csv_path)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=41)

dtclassifier = DecisionTreeClassifier(min_samples_split=2, max_depth=1)
dtclassifier.fit(x_train, y_train)

y_pred = dtclassifier.predict(x_test)
print(accuracy_score(y_test, y_pred))


'''
A tanítás során adódó komplikációkat csak az NJCleanerben előbukkanó apróbb hibák okozták, ezek kijavítása után gondtalanul működött minden.
A fiteléseknél a min_samples_split-et 1 és 3 között, a max_depth-et pedig 1 és 4 között változtattam.
Észrevétel:a min_samples_split nem befolyáslja az accuracyt, csak a max_depth, a legjobb eredményt akkor kaptam, mikor ez 4 volt.


min_samples_split, max_depth, accuracy:
1, 4, 0.7849166666666667
1, 3, 0.7839166666666667
1, 2, 0.7823333333333333
1, 1, 0.7773333333333333
2, 4, 0.7849166666666667
2, 3, 0.7839166666666667
2, 2, 0.7823333333333333
2, 1, 0.7773333333333333
3, 4, 0.7849166666666667
3, 3, 0.7839166666666667
3, 2, 0.7823333333333333
3, 1, 0.7773333333333333
'''

