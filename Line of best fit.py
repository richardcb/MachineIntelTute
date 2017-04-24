from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# create data set
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# write out formula of slope (m)
# **2 = ^2
# REMEMBER PEMDAS (peren, exp, multi, div, add, sub)
def best_fit_slope(xs,ys):
    m = ( ( (mean(xs) * mean(ys)) - mean(xs*ys) ) /
          ((mean(xs) **2) - mean(xs**2)))
    return m

m = best_fit_slope(xs,ys)

print(m)
