import numpy as np
from tqdm import tqdm

def std_sigmoid(x):
    mu = np.mean(x)
    std = np.std(x)
    transform = (x - mu) / (std)
    return 1 / (1 + np.exp(-1 * transform))

def cluster_avg_1D(cluster_data, y_data):
    """Average out the y_data in each cluster,
    to use as y-axis positions for the graph visualization"""
    clusters = np.unique(cluster_data)
    avg_arr = np.zeros(np.shape(clusters))
    i = 0
    for cluster in clusters:
        if cluster == -2:
            continue
        cl_idx = (cluster_data == cluster).nonzero()
        sum_ = 0
        for val in y_data[cl_idx]:
            sum_ += val
        sum_ /= np.size(cl_idx)
        avg_arr[i] = sum_
        i += 1

    return avg_arr


def cluster_most_common(cluster_data, y_data):
    """Get the most common y_data val in each cluster"""
    clusters = np.unique(cluster_data)
    most_arr = np.zeros(np.shape(clusters), dtype=int)
    i = 0

    for cluster in clusters:
        if cluster == -2:
            continue
        cl_idx = (cluster_data == cluster).nonzero()
        values, counts = np.unique(y_data[cl_idx], return_counts=True)
        most_ = values[np.argmax(counts)]
        most_arr[i] = int(most_)
        i += 1

    return most_arr


def epsilon_balls(data, epsilon):
    """Return (distances, indices) of points in B(r,x)"""
    distances = []
    indices = []
    for x in tqdm(data):
        d = np.linalg.norm(x - data, axis=1)
        idx = (d < epsilon).nonzero()
        dist = d[idx]
        distances.append(dist)
        indices.append(idx)
    return distances, indices

def compute_cluster_yaxis(clusters, semantic_dist, func=cluster_avg_1D):
    y_data = []
    for tslice in clusters:
        y_datum = func(tslice, semantic_dist)
        y_data.append(y_datum)

    return y_data

def cosine_window(distance, width=1):
    """ Returns weights for smoothing the Morse density """
    mask = np.abs(distance) <= width
    return(1/2)*(1+np.cos(np.pi*distance/width))*mask

def compute_point_rates(data, time, distances, width):
    lambdas = np.zeros(np.size(time))
    for i, d in tqdm(enumerate(distances), desc="Computing f-rates"):
        t0 = time[i]
        idx = (d <= width).nonzero()[0]
        vals_in_series = time[idx]
        vals_in_series.sort()
        t0_index = np.squeeze(np.where(vals_in_series == t0))
        deltas = np.diff(vals_in_series)
        np.roll(deltas, -t0_index)
        N = np.size(deltas)
        time_weights = np.zeros(N)
        for k, _ in enumerate(deltas):
            time_weights[k] = min([k, np.abs(k - (N - 1))])
        time_weights = np.exp(-time_weights)
        if np.size(idx) == 1:
            lambdas[i] = np.inf
        else:
            lambdas[i] = np.average(deltas, weights=time_weights)

    return lambdas


def weighted_clusters(
    data,
    time,
    checkpoints,
    densities,
    clusterer,
    kernel,
    overlap,
    kernel_params=None,
    eps=0.01,
):
    # -2 as a placeholder for "not clustered"
    clusters = np.ones((np.size(checkpoints), np.size(time)), dtype=int) * -2
    weights = np.zeros((np.size(checkpoints), np.size(time)))

    cp_with_ends = [np.amin(time)] + list(checkpoints) + [np.amax(time)]
    for idx, t0 in enumerate(checkpoints):
        bin_width = (cp_with_ends[idx + 2] - cp_with_ends[idx]) / 2
        bin_width *= 1 / (2 - overlap)
        if kernel_params == None:
            for i in np.arange(np.size(time)):
                weights[idx, i] = kernel(
                    t0,
                    time[i],
                    densities[i],
                    bin_width,
                )
        else:
            for i in np.arange(np.size(time)):
                weights[idx, i] = kernel(
                    t0,
                    time[i],
                    densities[i],
                    bin_width,
                    params=kernel_params,
                )
        slice_ = (weights[idx] >= eps).nonzero()
        slice_ = np.squeeze(slice_)
        data_slice = data[slice_]
        if data_slice.shape[0]==0:
            clusters[idx, slice_] = -2
            continue
        if data[slice_].ndim == 1:
            data_slice = data_slice.reshape(-1, 1)

        if ((weights < 1) & (0 < weights)).any():
            try:
                cluster_labels = clusterer.fit(
                    data_slice, sample_weight=weights[idx, slice_]
                ).labels_
            except:
                print(
                    "Clusterer does not accept sample weights. Falling back to unweighted clustering."
                )
                cluster_labels = clusterer.fit(data_slice).labels_
        else:
            cluster_labels = clusterer.fit(data_slice).labels_

        clusters[idx, slice_] = cluster_labels

    return clusters, weights