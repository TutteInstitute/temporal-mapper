import networkx as nx
import numpy as np
from scipy.stats import multinomial

def topic_contract(mapper, v):
    G = mapper.G.to_undirected()
    neighbours = G.neighbors(v)
    d = len([n for n in neighbours])
    if d == 0:
        return None
    if (mapper.G.in_degree(v) == 1):
        single_edge_contract(mapper.G, v)
    if (d==1) and (mapper.G.out_degree(v) == 1):
        # this is v a source
        return None
    idxs = nx.get_node_attributes(G, 'slice_no')
    idx = idxs[v]
    w = mapper.weights[idx,:]
    sizes = np.array(
        [np.sum(w[mapper.get_vertex_data(u)]) for u in neighbours]
    )
    sizes /= np.sum(sizes)
    impurity = 1 - np.sum(sizes**2)
    topic = nx.get_node_attributes(G, 'topic')
    if impurity <= 1/(2*d):
        # highly homogeneous, merge node into most similar
        s=0
        best = v
        for u in neighbours:
            if sizes[u]>s:
                s=sizes[u]
                best = u
        topic[v] = topic[best]
    else:
        pass
    nx.set_node_attributes(G, topic, 'topic')
         
def single_edge_contract(G, v):
    # special case to check for large semantic drift.
    drift = nx.get_edge_attributes(G, 'drift')
    drifts = [drift[e] for e in G.edges()]
    sigma = np.std(drifts)
    # this line is stupid but I can't figure out
    # how else to get just one edge out of an OutEdgeDataView
    e = [e for e in G.in_edges(v)][0] 
    if drift[e] < 2*sigma:
        topic = nx.get_node_attributes(G, 'topic')
        topic[v] = topic[e[0]]
        nx.set_node_attributes(G, topic, 'topic')