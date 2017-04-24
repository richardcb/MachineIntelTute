import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

style.use('ggplot')

# import data into data frame
df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

# create method to convert non-numerical data
# to numerical data in order for clustering
# method to work correctly


def handle_non_numerical_data(df):
    columns = df.columns.values
    # loop through each column
    for column in columns:
        # emppty the dictionary
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]
        # find whether column is integer or not
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # if not integer then convert to list
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            # take the unique elements and populate the dictionary
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            # reset values of df[column] by mapping the convert_to_int function
            # to the value within that column
            df[column] = list(map(convert_to_int, df[column]))
    return df
df = handle_non_numerical_data(df)
#print(df.head())

# select data to drop in order to test outcome %
# ['pclass'], ['survived'], ['age'], ['sibsp'],
# ['parch'], ['ticket'], ['fare'], ['cabin'], ['embarked']
# ['home.dest']
df.drop(['boat', 'sex'], 1, inplace=True)

# define features and labels
X = np.array(df.drop(['survived'], 1).astype(float))
# scale the data
X = preprocessing.scale(X)
y = np.array(df['survived'])

# define classifier
clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))
