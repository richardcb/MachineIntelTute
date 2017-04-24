import numpy as np
import pandas as pd
import quandl, math, datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# high minus low percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

# percent change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


# output df (things we want to see)
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# math.ceil will take any value and get to ceiling
# ie a df len is equal to 0.1. math.ceil will round that up to 1 (or nearest whole number)
forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)

# new label we will be focusing on, shift 30 days into future
df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())

# x = features
# y = label
# df.drop returns a new df
x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
# x_lately is the last 30 days of data
x_lately = x[-forecast_out:]
x = x[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

# # setup classifier "note: clf = x" x can be any method, ie svm
# clf = LinearRegression(n_jobs=-1)
# # trains the classifier
# clf.fit(x_train, y_train)
# # save the classifier
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)

#print(accuracy)

# this gives the prediction with scikitlearn
forecast_set = clf.predict(x_lately)

print(forecast_set)
print(accuracy)
print(forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# gives dates on the axes of graph
for i in forecast_set:
     next_date = datetime.datetime.fromtimestamp(next_unix)
     next_unix += one_day
     df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df ['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

