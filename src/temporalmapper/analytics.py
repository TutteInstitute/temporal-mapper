import networkx as nx
from math import isnan
import pandas as pd
import numpy as np
import json 
from temporalmapper.utilities import (
    prepare_plotly_graph_objects,
    compute_time_semantic_positions
)

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
    for node in mapper.G.nodes():
        if node.split(':')[0]==str(idx):
            nodes.append(node)
    return nodes
    
def slice_df(mapper, idx):
    G = mapper.G
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
    G = mapper.G
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

def initial_y_position(self):
    """ Compute initial positions for the y-axis of temporal plot """
    if self.n_components == 1:
        y_initial_pos = self.data[:,0]
    if self.n_components == 2:
        y_initial_pos = np.arctan2(self.data[:,1], self.data[:,0])
    else:
        pca = PCA(n_components=1)
        y_initial_pos = pca.fit_transform(self.data)
    return y_initial_pos

def get_previous_node(G,node):
    in_nodes = [u for (u,v) in G.in_edges(node)]
    if len(in_nodes)!=1:
        raise ValueError
    else:
        u = in_nodes[0]
    return u

def static_topics(mapper):
    G = mapper.G
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
    centroids = nx.get_node_attributes(mapper.G, 'centroid')
    counts = nx.get_node_attributes(mapper.G, 'count')
    time = nx.get_node_attributes(mapper.G, 'median_time')
    cluster_labels = nx.get_node_attributes(mapper.G, 'topic_name')
    if {} in [centroids,counts,time]:
        mapper.populate_node_attrs()
        centroids = nx.get_node_attributes(mapper.G, 'centroid')
        counts = nx.get_node_attributes(mapper.G, 'count')
        time = nx.get_node_attributes(mapper.G, 'median_time')

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
        'topic_name':cluster_labels[topic[0]],
        'final_node':topic[0],
        #'centroid':topic_centroid,
        'count':int(topic_count), 
        'time_span':topic_time_span,
        'nodes':[str(s) for s in topic],
    }
    return topic_summary

def export_chronoscope(mapper, filepath='chronoscope_data.json'):
    growth_data = pd.concat([
        slice_df(mapper, idx) for idx in range(mapper.N_checkpoints)
    ]).to_dict(orient='records')
    shifts = top_shifts(mapper).to_dict(orient='records')
    
    slice_labels = {idx:int(label) for idx,label in enumerate(mapper.checkpoints)}

    positions = nx.get_node_attributes(mapper.G,'ts_pos')
    if len(positions)==0:
        positions = compute_time_semantic_positions(
            mapper,
            initial_y_position(mapper),
            layout_optimization = 'barycenter',
        )

    cluster_labels = {}
    hover_text = {}
    if len(hover_text)==0:
        # construct some default hover text.
        for node in mapper.G.nodes():
            idx = mapper.get_vertex_data(node)
            median_time = np.median(mapper.time[idx])
            node_name = cluster_labels.get(node,'')
            label_str = f"{node_name}<br>Node {node}<br>Time: {median_time}"
            hover_text[node] = label_str
    
    edge_trace, node_trace = prepare_plotly_graph_objects(
        mapper,
        positions,
        hover_text=hover_text,
        edge_scaling = 1,
        node_scaling = 1,
        node_size_bounds = (5,50),
        edge_weight_bounds = (0.1,1),
        node_size_scale = 'sigmoid',
    )
    # Convert traces to JSON separately
    edge_json = [trace.to_plotly_json() for trace in edge_trace]
    node_json = node_trace.to_plotly_json()

    # Topic summaries
    static_topics = static_topics(mapper)
    topic_summaries = {key:static_topic_summary(mapper, topic) for key,topic in static_topics(mapper).items()}
    
    config = {
        "slice_labels": slice_labels
    }
    
    json_data = {
        'config':config,
        'growth_data':growth_data,
        'shifts':shifts,
        'network-traces':{'node':node_json, 'edge':edge_json}
    }
    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=2)

