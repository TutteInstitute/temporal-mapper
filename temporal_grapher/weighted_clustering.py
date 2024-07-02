import fast_hdbscan ## This is the modified local copy!

import sys
import numpy as np
import pandas as pd
import math
import numba
from warnings import warn

def gaussian(t0, t, density, binwidth, epsilon = 0.01, params=None):
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
    if np.abs(distance) < width:
        return (1/2)*(1+np.cos(np.pi*distance/width))
    else:
        return 0

def compute_point_rates(data, time, distances, width, sensitivity=1):
    d_max = width/np.size(time)
    lambdas = np.zeros(np.size(time))
    for i,d in enumerate(distances):
        t0 = time[i]
        idx=(d<=d_max).nonzero()[0]
        vals_in_series = time[idx]
        vals_in_series.sort()
        t0_index = np.where(vals_in_series == t0)[0][0]
        deltas=np.diff(vals_in_series)
        np.roll(deltas, -t0_index)
        N=np.size(deltas)
        time_weights = np.zeros(N)
        for k, _ in enumerate(deltas):
            time_weights[k] = min(
                [k, np.abs(k-(N-1))]
            )
        time_weights = np.exp(-time_weights)
        if np.size(idx)==1:
            lambdas[i] = 0
        else:
            lambdas[i] = np.average(deltas, weights=time_weights)
    iso_idx = (lambdas == 0).nonzero()
    # apply the window:
    smoothed_lambdas = np.zeros(np.size(time))
    for j, d in enumerate(distances):
        val = 0
        norm = 0
        idx=(d<=25*d_max).nonzero()[0]
        for i in idx:
            val += window(d[i], 5*d_max)*lambdas[i]
            norm +=  window(d[i], 5*d_max)
        smoothed_lambdas[j] = val/norm

    #smoothed_lambdas = lambdas
    iso_idx = (smoothed_lambdas == 0)
    rates = smoothed_lambdas 
    rates[iso_idx] = np.inf
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
        try:
            cluster_labels = clusterer.fit(data[slice_], sample_weight=weights[idx,slice_]).labels_
        except(ValueError):
            data_reshape = data[slice_].reshape(-1,1)
            cluster_labels = clusterer.fit(data_reshape, sample_weight=weights[idx,slice_]).labels_
        clusters[idx, slice_] = cluster_labels
    
    return clusters, weights
