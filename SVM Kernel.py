import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

# read in csv data set
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# replace known missing data (?)
# note: most classifiers treat '99999' as an outlier
df.replace('?', -99999, inplace=True)
# remove unwanted column
df.drop(['id'], 1, inplace=True)

# cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# define the classifier and fit
clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

