import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os, sys
from scipy import stats
import numpy as np

data = pd.read_csv('../user_data.csv')

df = data[["number", "time"]]

#print(data)

# scatterplot of inputs data
plt.scatter(df["number"], df["time"])

# scatterplot of inputs data
plt.scatter(df["number"], df["time"])

# create arrays
X = df.values

for index, line in enumerate(open('../user_data.csv', 'r').readlines()):
    w = line.split(' ')
    l1 = w[1:8]
    l2 = w[8:15]

    try:
        list1 = map(float, l1)
        list2 = map(float, l2)
    except ValueError:
        print ('Line {i} is corrupt!'.format(i = index))
        break

    result = stats.ttest_ind(list1, list2)
    print (result[1])
# instantiate model
nbrs = NearestNeighbors(n_neighbors = 3)
# fit model
#nbrs.fit(X)