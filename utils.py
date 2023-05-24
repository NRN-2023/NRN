import numpy as np
import pandas as pd
from py import test
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

def load_data(data, K, scaler):
    data = np.copy(data)
    _data = scaler.transform(data)
    #print(_data)
    dist_m = pairwise_distances(_data, _data, metric='nan_euclidean')
    col_counts = []
    for col in data.T:
        col_counts.append(np.count_nonzero(~np.isnan(col)))
    col_counts = np.array(col_counts)
    sum_col_counts = np.sum(col_counts)

    p_m = np.zeros((dist_m.shape))
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            p_m[i][j] = p_m[j][i] = np.sum(col_counts[np.where(np.logical_or(np.isnan(data[i]), np.isnan(data[j])))[0]]) / sum_col_counts
            
    alpha = 0.15
    dmax = np.nanmax(dist_m)
    dist_m /= dmax
    dist_m = dist_m * (1 - alpha) + alpha * p_m

    indices = np.argsort(dist_m, axis=1)[:, 1:K+1]
    neighbors = np.take(_data, indices, axis=0)
    train_data = np.concatenate((_data, neighbors.reshape(neighbors.shape[0], -1)), axis=1)

    return train_data

def generateMissing(org_data, missing_ratio, seed=None):
    np.random.seed(seed)
    data = org_data.copy()
    shape = data.shape
    n = int(missing_ratio * data.size)
    indexes = np.random.choice(data.size, n, replace=False)
    i, j = np.unravel_index(indexes, shape)
    data = np.array(data)
    data[i, j] = np.nan
    return data