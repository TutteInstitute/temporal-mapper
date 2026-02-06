import numpy as np
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

import temporalmapper as tm
import temporalmapper.utilities as tmutils
import temporalmapper.weighted_clustering as tmwc

def computeGraph(kwargs={}):
    """ Integration test from loading data to producing a graph. """
    data_time = np.load(data_folder+"genus1_test.npy")
    timestamps_unsort = data_time[:,0].T
    sorted_indices = np.argsort(timestamps_unsort)
    data = data_time[sorted_indices]
    timestamps = timestamps_unsort[sorted_indices]
    N_data = np.size(timestamps) 
    clusterer = DBSCAN()
    TM = tm.TemporalMapper(
        timestamps,
        data,
        clusterer,
        verbose=True,
        N_checkpoints = 24,
        neighbours = 50,
        slice_method='time',
        overlap = 0.5,
        rate_sensitivity=1,
        kernel=tmwc.square,
    )
    TM.build()
    return TM

data_folder = 'data/'
with open(data_folder+'TMTest.pkl', 'wb') as f:
    TM = computeGraph()
    pkl.dump(TM, f)
    print(f"Saved updated Temporal Mapper to {data_folder}TMTest.pkl")
