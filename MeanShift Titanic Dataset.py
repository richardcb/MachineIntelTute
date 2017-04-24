import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

style.use('ggplot')

# import data into data frame
df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
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
clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

# iterate through the labels and populate the data
for i in range(len(X)):
    # use iloc to reference the index of the df, aka the row of the df
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'] == float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived'] == 1) ]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate
print(survival_rates)
