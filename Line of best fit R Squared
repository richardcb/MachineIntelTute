from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

#define the style used for plot
style.use('fivethirtyeight')

# create data set
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# write out formula of line of best fit (m & b)
# **2 = ^2
# REMEMBER PEMDAS (peren, exp, multi, div, add, sub)
def best_fit_slope_and_intercept(xs, ys):
    m = ( ( (mean(xs) * mean(ys)) - mean(xs*ys) ) /
          ((mean(xs) ** 2) - mean(xs** 2)))
    b = mean(ys) - m * mean(xs)
    return m, b

# define m and b
m, b = best_fit_slope_and_intercept(xs, ys)

# define the actual line to be drawn with for loop
regression_line = [(m * x) + b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

# draw scatter plot and line
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()
