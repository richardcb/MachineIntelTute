import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

# set plot style
style.use('ggplot')

# define svm class
class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data
        # { ||w|| : [w,b] }
        opt_dict = {}
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
            self.max_feature_value = max(all_data)
            self.min_feature_value = min(all_data)
            #all_data = None
            # support vectors will be yi(xi.w+b) = 1
            # will know when optimum value is found when
            # both +tive and -tive classes have a value close to 1
            # note: how close to one is dynamic and can be set by user (modify step_size)
            step_sizes = [self.max_feature_value * 0.1,
                          self.max_feature_value * 0.01,
                          # point of expense
                          self.max_feature_value * 0.001,
                          # point of crazy
                          #self.max_feature_value * 0.0001
                          ]
            # extremely expensive (machine intensive)
            # can adjust number for increased/decreased accuracy on b
            b_range_multiple = 5
            # we don't need to take as small a steps
            # as we do with w
            b_multiple = 5
            # the first element in vector w
            latest_optimum = self.max_feature_value * 10
            for step in step_sizes:
                w = np.array([latest_optimum, latest_optimum])
                # we can do this because convex problem
                optimized = False
                while not optimized:
                    # note: can be run in parallel for speed - future work
                    for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                       self.max_feature_value * b_range_multiple,
                                       step * b_multiple):
                        for transformation in transforms:
                            w_t = w * transformation
                            found_option = True
                            # weakest link in the SVM fundamentally
                            # SMO attempts to fix this a bit
                            # constraint function = yi(xi.w+b) >= 1
                            # #### add a break here later for speed
                            for i in self.data:
                                for xi in self.data[i]:
                                    yi = i
                                    if not yi * (np.dot(w_t, xi) + b) >= 1:
                                        found_option = False
                            if found_option:
                                opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                    if w[0] < 0:
                        optimized = True
                        print('Optimized a step.')
                    else:
                        w = w - step
                # norms are the sorted magnitudes
                norms = sorted([n for n in opt_dict])
                # opt_choice is the magnitude in the first position
                # aka: the smallest magnitude
                opt_choice = opt_dict[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                latest_optimum = opt_choice[0][0] + step * 2
        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi * (np.dot(self.w, xi) + self. b))

    def predict(self, features):
        # sign( x. w + b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        # display hyperplane showing plots and SV's
        # hyperplane = x.w+b
        # we want v = x.w+b
        # care when:
        # Positive Supper Vector (PSV) = 1
        # Negative Support Vector (NSV) = -1
        # Decision Boundary Hyperplane (DECB) = 0

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]
        # limit the size of the hyperplane
        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        # PSV HP = (w.x+b) = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        # draw line
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # NSV HP = (w.x+b) = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        # draw line
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # DECB = (w.x+b) = 0
        decb1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        decb2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        # draw line
        self.ax.plot([hyp_x_min, hyp_x_max], [decb1, decb2], 'y--')

        plt.show()

# create data dictionary and keys
data_dict = {-1: np.array([[1, 7],
                          [2, 8],
                          [3, 8], ]),
             1: np.array([[5, 1],
                         [6, -1],
                         [7, 3]])}
svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8]]
for p in predict_us:
    svm.predict(p)

svm.visualize()
