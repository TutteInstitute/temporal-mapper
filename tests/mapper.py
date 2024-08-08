import numpy as np
import sys, os
import networkx as nx
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

sys.path.append(os.path.relpath("./../temporal-mapper"))
import temporal_mapper as tm
import utilities_ as tmutils
import weighted_clustering as tmwc

data_folder = 'data/'

def computeGraph(kwargs={}):
    """ Integration test from loading data to producing a graph. """
    data_arr = np.load(data_folder+'arxivML_test_data.npy')
    timestamps = data_arr[:,0]
    map_data = data_arr[:,1:]
    clusterer = DBSCAN()

    TM = tm.TemporalMapper(
        timestamps,
        map_data,
        clusterer,
        **kwargs,
    )
    TM.build()
    return 0

def centroidDatamap(kwargs={}):
    """ Unit test for utilities_.centroid_datamap """
    with open(data_folder+'TMTest.pkl', 'rb') as f:
        TM = pkl.load(f)
        f.close()
    tmutils.centroid_datamap(
        TM, **kwargs
    )
    return 0

def timeSemanticPlot(kwargs={}):
    """ Unit test for utilities.time_semantic_plot """
    with open(data_folder+'TMTest.pkl', 'rb') as f:
        TM = pkl.load(f)
        f.close()
    semantic_data = PCA(n_components=1).fit_transform(TM.data)
    tmutils.time_semantic_plot(
        TM, semantic_data, **kwargs,
    )
    return 0


def test_computeGraph():
    parameters = [
        {'N_checkpoints':8, 'slice_method':'time'},
        {'N_checkpoints':8, 'slice_method':'data'},
        {'N_checkpoints':8, 'kernel':tmwc.square, 'rate_sensitivity':0} # vanilla mapper
    ]
    for i in range(len(parameters)):
        assert computeGraph(kwargs=parameters[i]) == 0
        
def test_centroidDatamap():
    parameters = [
        {'bundle':False},
        {'bundle':True},
    ]
    for i in range(len(parameters)):
        assert centroidDatamap(kwargs=parameters[i]) == 0
        
def test_timeSemanticPlot():
    parameters = [
        {'bundle':False},
        {'bundle':True},
    ]
    for i in range(len(parameters)):
        assert timeSemanticPlot(kwargs=parameters[i]) == 0
    


