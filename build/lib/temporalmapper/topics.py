import networkx as nx
import numpy as np
from scipy.stats import multinomial

# BIC = nLn + d/2 log n
def multinomial_edge_contract(mapper, v):
    G = mapper.G
    neighbours = G.neighbours(v)
    idxs = nx.get_node_attributes(G, 'slice_no')
    idx = idxs[v]
    w = mapper.weights[idx,:]
    sizes = {
        u:np.sum(w[mapper.get_vertex_data(u)])
        for u in neighbours
    }
    obs_merged = [sizes[u] for u in neighbours]
    n =  np.size(mapper.get_vertex_data(v))
    rv_merged = multinomial(
        n, obs_merged/np.sum(obs_merged)
    )
    ll_merged = rv_merged.logpmf(
        [int(sizes[u]) for u in neighbours]
    )

    sizes[v] = np.sum(w[mapper.get_vertex_data(v)])
    obs_split = [sizes[u] for u in neighbours]
    rv_split = multinomial(
        n, obs_split/np.sum(obs_split)
    )
    ll_split= rv_split.logpmf(
        [int(sizes[u]) for u in neighbours]
    )

    d = len(neighbours)
    topic = nx.get_node_attributes(G, 'topic')
    if (-2*n*ll_split+(d+1)*np.log(n)) >= (-2*n*ll_merge+d*np.log(n)):
        # merge wins
        s=0
        best = v
        for u in neighbours:
            if sizes[u]>s:
                s=sizes[u]
                best = u
        topic[v] = u
    else:
        # split wins
        pass
    nx.set_node_attributes(G, topic, 'topic')
         
