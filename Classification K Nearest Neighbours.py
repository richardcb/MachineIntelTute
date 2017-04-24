import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

accuracies = []

# run test x times
for i in range(5):
    # read in csv data set
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')

    # replace known missing data (?)
    # note: most classifiers treat '99999' as an outlier
    df.replace('?', -99999, inplace=True)
    # remove unwanted column
    df.drop(['id'], 1, inplace=True)

    # X = features
    X = np.array(df.drop(['class'], 1))
    # y = labels
    y = np.array(df['class'])

    # cross validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    # define the classifier and fit
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    # print(accuracy)

    # prediction
    # example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
    #
    # example_measures = example_measures.reshape(len(example_measures), -1)
    #
    # prediction = clf.predict((example_measures))
    # print(prediction)

    accuracies.append(accuracy)

print(sum(accuracies) / len(accuracies))
