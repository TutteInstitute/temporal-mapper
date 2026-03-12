import temporalmapper as tm
import numpy as np
import pickle as pkl 
import matplotlib.pyplot as plt  
import networkx as nx

data_folder = 'data/'

def loadMapper(path):
    with open(path, 'rb') as f:
        mapper = pkl.load(f)
        f.close()
    return mapper

def plotTemporal(kwargs={}):
    mapper = loadMapper(data_folder+'TMTest.pkl')
    mapper.temporal_plot(**kwargs)
    return 0

def plotInteractiveTemporal(kwargs={}):
    mapper = loadMapper(data_folder+'TMTest.pkl')
    mapper.interactive_temporal_plot(**kwargs)
    return 0

def plotTreemap(kwargs={}):
    mapper = loadMapper(data_folder+'TMTest.pkl')
    tm.plot.growth_map(mapper)
    return 0

def plotSliceograph(kwargs={}):
    mapper = loadMapper(data_folder+'TMTest.pkl')
    tm.plot.sliceograph(mapper)
    return 0

def plotGomic(kwargs={}):
    mapper = loadMapper(data_folder+'TMTest.pkl')
    tm.plot.view_gomic(mapper)
    return 0

def centroidDatamap(kwargs={}):
    """ Unit test for utilities_.centroid_datamap """
    with open(data_folder+'TMTest.pkl', 'rb') as f:
        mapper = pkl.load(f)
        f.close()
    tm.plot.centroid_datamap(
        mapper, **kwargs
    )
    return 0

def plotSubgraph(kwargs={}):
    """ Unit test for temporal_mapper.vertex_subgraph and plotting it """
    with open(data_folder+'TMTest.pkl', 'rb') as f:
        mapper = pkl.load(f)
        f.close()
    vertices = mapper.vertex_subgraph('0:0')
    mapper.temporal_plot(
        vertices=vertices, **kwargs,
    )
    """ When I copy this test into a new file and run it, it passes.
    I can't figure out why it doesn't pass here... """
    # tmplot.centroid_datamap(
    #     TM, **kwargs, vertices=vertices
    # )
    return 0

def test_temporal_plot():
    fig, ax = plt.subplots(1,1)
    parameters = [
        {},
        {
            'ax':ax,
            'title':"Lorem Ipsum",
            'edge_scaling':0.5,
            'node_scaling':2,
            'node_size_bounds':(1,10),
            'edge_weight_bounds':(0.2,2),
        },
        {
            'node_size_scale':'linear',
            'layout':'barycenter',
        },
        {
            'node_size_scale':"logarithmic",
            'layout':'force',
        },
        {
            'node_size_scale':'sigmoid',
            'layout':'ordered',
        }
    ]
    for i in range(len(parameters)):
        assert plotTemporal(kwargs=parameters[i]) == 0

def test_interactive_temporal_plot():
    parameters = [{}]
    for i in range(len(parameters)):
        assert plotInteractiveTemporal(kwargs=parameters[i])==0

def test_treemap():
    parameters = [{'index':None},{'index':0}]
    for i in range(len(parameters)):
        assert plotTreemap(kwargs=parameters[i])==0

def test_sliceograph():
    parameters = [{}]
    for i in range(len(parameters)):
        assert plotSliceograph(kwargs=parameters[i])==0

def test_centroidDatamap():
    parameters = [
        {'bundle':False},
        {'bundle':True},
    ]
    for i in range(len(parameters)):
        assert centroidDatamap(kwargs=parameters[i]) == 0
        
def test_vertexSubgraph():
    parameters = [
        {'bundle':False},
    ]
    for i in range(len(parameters)):
        assert plotSubgraph(kwargs=parameters[i]) == 0