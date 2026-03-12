import networkx as nx
from math import isnan
import pandas as pd
import numpy as np
import json 

def sources_and_sinks(G):
    source_nodes = [
        node for node in G.nodes() if G.in_degree(node)==0
    ]
    sink_nodes = [
        node for node in G.nodes() if G.out_degree(node)==0
    ]   
    return source_nodes, sink_nodes

def splits_and_merges(G):
    splits = []
    merges = []
    for node in G.nodes():
        if (G.in_degree(node)==1) and (G.out_degree(node)>1):
            splits.append(node)
        elif (G.out_degree(node)==1) and (G.in_degree(node)>1):
            merges.append(node)
    return splits, merges

def compute_growth(G):
    counts = nx.get_node_attributes(G, 'count')
    growth = {}
    for node in G.nodes():
        in_sizes = [counts[u] for (u,v) in G.in_edges(node)]
        in_size = sum(in_sizes)
        if in_size != 0:
            growth[node] = (counts[node]-in_size)/in_size
        else:
            growth[node] = np.nan
    return growth

def nodes_in_slice(mapper, idx):
    nodes = []
    for node in mapper.graph.nodes():
        if node.split(':')[0]==str(idx):
            nodes.append(node)
    return nodes
    
def slice_df(mapper, idx):
    G = mapper.graph
    counts = nx.get_node_attributes(G, 'count')
    growth = compute_growth(G)
    
    nodes = nodes_in_slice(mapper, idx)
    top_n = 5

    count_values = np.array([
        counts[v] for v in nodes    
    ])
    growth_values = np.array([
        growth[v] for v in nodes
    ])
    slice_df = pd.DataFrame({
        'node':nodes,
        'slice':[idx]*len(nodes),
        'count':count_values,
        'growth':growth_values,
    })
    return slice_df
    
def semantic_shift(G):
    centroids = nx.get_node_attributes(G, 'centroid')
    centroids = {
        node:np.array(centroids[node]) for node in centroids.keys()
    }
    shift = {}
    for node in G.nodes():
        if G.in_degree(node)==1:
            (u,v) = [e for e in G.in_edges(node)][0]
            shift[node] = np.linalg.norm(centroids[u]-centroids[node])
        else:
            shift[node] = np.nan
    return shift

def top_shifts(mapper, topN=10):
    G = mapper.graph
    shifts = semantic_shift(G)
    non_nan_shifts = []
    non_nan_nodes = []
    previous_nodes = []
    for node in shifts.keys():
        if not isnan(shifts[node]):
            non_nan_nodes.append(node)
            non_nan_shifts.append(shifts[node])
            (u,v) = [e for e in G.in_edges(node)][0]
            previous_nodes.append(u)
    
    non_nan_shifts = np.array(non_nan_shifts)
    non_nan_nodes = np.array(non_nan_nodes)
    previous_nodes = np.array(previous_nodes)
    mean = np.mean(non_nan_shifts)
    std = np.std(non_nan_shifts)

    sig_idx = non_nan_shifts>mean+std
    df = pd.DataFrame({
        'node': non_nan_nodes[sig_idx],
        'shift': non_nan_shifts[sig_idx],
        'previous':previous_nodes[sig_idx],
    })
    top = df.reindex(df['shift'].abs().nlargest(topN).index)
    return top

def get_previous_node(G,node):
    in_nodes = [u for (u,v) in G.in_edges(node)]
    if len(in_nodes)!=1:
        raise ValueError
    else:
        u = in_nodes[0]
    return u

def static_topics(mapper):
    G = mapper.graph
    #topic_enders = [n for n in G.nodes() if (G.degree(n) != 2) and (G.in_degree(n) == 1)]
    sources, sinks = sources_and_sinks(G)
    static_topics = {}
    
    for s in sinks:
        ancestors = [s]
        visited = {s}
        queue = [s]
        
        while queue:
            current = queue.pop(0)
            predecessors = list(G.predecessors(current))
            
            for pred in predecessors:
                if pred in visited:
                    continue
                #if pred in topic_enders:
                #    continue
                
                ancestors.append(pred)
                visited.add(pred)
                queue.append(pred)
        
        static_topics[s] = ancestors
    
    return static_topics

def static_topic_summary(mapper, topic):
    centroids = nx.get_node_attributes(mapper.graph, 'centroid')
    counts = nx.get_node_attributes(mapper.graph, 'count')
    time = nx.get_node_attributes(mapper.graph, 'median_time')
    cluster_labels = nx.get_node_attributes(mapper.graph, 'topic_name')
    if {} in [centroids,counts,time]:
        mapper.populate_node_attrs()
        centroids = nx.get_node_attributes(mapper.graph, 'centroid')
        counts = nx.get_node_attributes(mapper.graph, 'count')
        time = nx.get_node_attributes(mapper.graph, 'median_time')

    topic_centroid = np.average([centroids[s] for s in topic], axis=0, weights=[counts[s] for s in topic])
    topic_count = np.sum([counts[s] for s in topic])
    times = np.array([time[s] for s in topic])
    # numpy types are not json serializable
    topic_time_span = {
        'min':float(np.amin(times)),
        'max':float(np.amax(times)),
        'median':float(np.average(times, weights=[counts[s] for s in topic])),
    }
    topic_summary = {
        'topic_name':cluster_labels.get(topic[0], ""),
        'final_node':topic[0],
        #'centroid':topic_centroid,
        'count':int(topic_count), 
        'time_span':topic_time_span,
        'nodes':[str(s) for s in topic],
    }
    return topic_summary

