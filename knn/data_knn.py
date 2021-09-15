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


# instantiate model
nbrs = NearestNeighbors(n_neighbors = 3)
# fit model
nbrs.fit(X)

