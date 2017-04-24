import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

# define nearest neighbours function
def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data [group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    #print(vote_result, confidence)

    return vote_result, confidence

accuracies = []

# run test x times
for i in range(25):
    # read in csv dataset file
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    # replace known missing data (?)
    # note: most classifiers treat '99999' as an outlier
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    # convert entire dataset to float type
    full_data = df.astype(float).values.tolist()

    # randomise entire dataset
    random.shuffle(full_data)

    # define test size
    test_size = 0.4
    # create dictionaries
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    # all data up to last 20%
    train_data = full_data[:-int(test_size * len(full_data))]
    # last 20% of data
    test_data = full_data[-int(test_size * len(full_data)):]

    # populate dictionaries
    for i in train_data:
        # -1 index is the last index in the row of data ('class')
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        # -1 index is the last index in the row of data ('class')
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    print('Accuracy:', correct/total)
    accuracies.append(correct/total)

print('Overall Accuracy:', sum(accuracies) / len(accuracies))