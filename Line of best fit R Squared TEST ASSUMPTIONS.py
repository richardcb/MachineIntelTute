from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

#define the style used for plot
style.use('fivethirtyeight')

# create data set
#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# create data set
# hm = how many
def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# write out formula of line of best fit (m & b)
# **2 = ^2
# REMEMBER PEMDAS (peren, exp, multi, div, add, sub)
def best_fit_slope_and_intercept(xs, ys):
    m = ( ( (mean(xs) * mean(ys) ) - mean(xs*ys) ) /
          ( (mean(xs) ** 2) - mean(xs** 2) ) )
    b = mean(ys) - m * mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    # will calculate the squared error
    return sum((ys_line - ys_orig) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    # will calculate the mean of y for every y value present
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    # define the squared error of regression line
    squared_error_regr = squared_error(ys_orig, ys_line)
    # define the squared error of the mean of y line
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    # perform the coefficient of determination
    return 1 - (squared_error_regr / squared_error_y_mean)

# use this to change the features of the dataset
# note: how much, variance, step, pos or neg corr
xs, ys = create_dataset(40, 40, 2, correlation = 'pos')

# define m and b
m, b = best_fit_slope_and_intercept(xs, ys)

# define the actual line to be drawn with for loop
regression_line = [(m * x) + b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

# define R Squared
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# draw scatter plot and line
plt.scatter(xs, ys)
# shows how line of regression can predict a value beyond the data set
plt.scatter(predict_x, predict_y, s = 100, color='r')
plt.plot(xs, regression_line)
plt.show()
