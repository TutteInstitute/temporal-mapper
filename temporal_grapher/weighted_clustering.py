import fast_hdbscan ## This is the modified local copy!

import sys
import numpy as np
import pandas as pd
import math
import numba
from warnings import warn
from sklearn.neighbors import NearestNeighbors

def gaussian(t0, t, density, binwidth, epsilon = 0.1):
    K = -np.log(epsilon)/((binwidth/2)**2) 
    return np.exp(-K*(density*(t-t0))**2)

def square(t0, t, density, binwidth, epsilon = 0.1, params=(1,)):
    if params == None:
        print("Warning: Your kernel has parameters but you didn't pass any.")
    overlap, = params
    distance = (t-t0)
    out = (np.abs(distance)<((1+overlap)*binwidth/2)).astype(int)
    return out
    
def window(distance, width=1):
    # default to 10 because UMAP 
    if np.abs(distance) < width:
        return (1/2)*(1+np.cos(np.pi*distance/width))
    else:
        return 0

def compute_point_rates(data, time, k=250, width=1, sensitivity=1):
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nbrs.kneighbors(data)
    avg_knn_dist = np.min(distances[:,-1])
    lambdas = np.zeros(np.size(time))
    for i,d in enumerate(distances):
        idx=(d<=avg_knn_dist).nonzero()[0]
        vals_in_series = time[indices[i,idx]]
        vals_in_series.sort()
        if np.size(indices[i,idx])==1:
            lambdas[i] = 0
        else:
            lambdas[i] = np.mean(np.diff(vals_in_series))
    # apply the window:
    smoothed_lambdas = np.zeros(np.size(time))
    for idx, pt in enumerate(data):
        val = 0
        val += lambdas[idx]
        for i in range(k):
            val += window(distances[idx][i], width)*lambdas[indices[idx][i]]
        smoothed_lambdas[idx] = val
    zero_idx = (smoothed_lambdas == 0)
    nisolated = np.size((zero_idx).nonzero())
    if nisolated != 0:
        print(f'Warning: You have {nisolated} isolated points. If this is a small number, its probably fine.')
    rates = 1/smoothed_lambdas
    if sensitivity == -1:
        rates = np.log10(rates)
    else:
        rates = rates**sensitivity
    rates /= np.amax(rates[~zero_idx])
    rates[zero_idx] = 1
    return rates

def weighted_clusters(
    data,
    time,
    checkpoints,
    densities,
    clusterer,
    kernel,
    kernel_params=None,
    eps = 0.01
):
    # -2 as a placeholder for "not clustered"
    clusters = np.ones((np.size(checkpoints), np.size(time)),dtype=int)*-2
    weights =  np.zeros((np.size(checkpoints), np.size(time)))
                                     
    cp_with_ends = [np.amin(time)]+list(checkpoints)+[np.amax(time)]
    for idx, t in enumerate(checkpoints):
        bin_width = (cp_with_ends[idx+2]-cp_with_ends[idx])/2
        if kernel_params == None:
            for i in np.arange(np.size(time)):
                weights[idx,i] = kernel(
                    t, time[i], densities[i], bin_width,
                )
        else:
            for i in np.arange(np.size(time)):
                weights[idx,i] = kernel(
                    t, time[i], densities[i], bin_width, params=kernel_params,
                )
        slice_ = (weights[idx] >= eps).nonzero()
        slice_ = np.squeeze(slice_)
        cluster_labels = clusterer.fit(data[slice_], sample_weight=weights[idx,slice_]).labels_
        clusters[idx, slice_] = cluster_labels
    
    return clusters, weights
