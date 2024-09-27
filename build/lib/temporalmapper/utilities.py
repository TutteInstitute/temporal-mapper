import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from vectorizers.transformers import InformationWeightTransformer
from vectorizers import NgramVectorizer
from tqdm import tqdm, trange
from matplotlib.colors import to_rgba, rgb_to_hsv, hsv_to_rgb
from datashader.bundling import hammer_bundle
from pandas import DataFrame, concat

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


def graph_to_holoviews(G, dataset_func=None):
    """Take TemporalGraph.G and output the required HoloViews objects for a modified Sankey diagram."""
    nxNodes = G.nodes()
    nodes = nxNodes  # lol
    cnt = 0
    orphans = []
    idx = 0
    for node in nxNodes:
        if G.degree(node) == 0:
            cnt += 1
            orphans.append(node)
            continue
        G.nodes()[node]["index"] = idx
        idx += 1

    for node in orphans:
        G.remove_node(node)
    nxNodes = G.nodes()
    if cnt != 0:
        print(f"Warning: removed {cnt} orphan nodes from the graph.")
    nodes_ = {"index": [], "size": [], "label": [], "colour": [], "column": []}
    for i, node in enumerate(nxNodes):
        nodes_["index"].append(i)
        nodes_["size"].append(nodes[node]["count"])
        try:
            nodes_["label"].append(nodes[node]["label"])
        except KeyError:
            nodes_["label"].append(nodes[node]["index"])
        nodes_["colour"].append("#ffffff")
        nodes_["column"].append(nodes[node]["slice_no"])

    cmap = {nodes[node]["index"]: nodes[node]["colour"] for node in nodes}
    try:
        nodes = hv.Dataset(nodes_, "index", ["size", "label", "colour", "column"])
    except NameError:
        nodes = dataset_func(nodes_, "index", ["size", "label", "colour", "column"])

    edges = []

    for u, v, d in G.edges(data=True):
        uidx = nxNodes[u]["index"]
        vidx = nxNodes[v]["index"]
        u_size = nxNodes[u]["count"]
        v_size = nxNodes[v]["count"]
        edges.append((uidx, vidx, (u_size * d["src_weight"], v_size * d["dst_weight"])))

    return nodes, edges, cmap


def compute_cluster_yaxis(clusters, semantic_dist, func=cluster_avg_1D):
    y_data = []
    for tslice in clusters:
        y_datum = func(tslice, semantic_dist)
        y_data.append(y_datum)

    return y_data


def generate_keyword_labels(word_bags, TG, ngram_vectorizer=None, n_words=3, sep=" "):
    """Using a bag of words corresponding to each data point, get highly informative
    keywords for each cluster"""
    if ngram_vectorizer is None:
        ngram_vectorizer = NgramVectorizer()
        ngram_vectors = ngram_vectorizer.fit_transform(word_bags)
    else:
        ngram_vectors = ngram_vectorizer.transform(word_bags)
    ## Building cluster labels (crudely)
    IWT = InformationWeightTransformer()
    keywords = []
    for i in trange(len(TG.slices)):
        # build a vector for each cluster by summing the vectors of its constituent data
        cluster_vectors = []
        for cl in np.unique(TG.clusters[i]):
            if (cl == -1) or (cl == -2):
                # skip outliers
                continue
            cl_idx = (TG.clusters[i] == cl).nonzero()
            vectors_in_cluster = ngram_vectors[cl_idx]
            cl_vector = np.sum(vectors_in_cluster, axis=0)
            cluster_vectors.append(cl_vector)
        # print("IWT on slice:",i,end="\r")

        # IWT the vectors and get the most important keywords
        cluster_vectors = np.squeeze(np.array(cluster_vectors))
        weighted_vectors = IWT.fit_transform(cluster_vectors)
        cluster_keywords = []
        for cl_vector in weighted_vectors:
            cl_vector = np.squeeze(cl_vector)
            highest = np.argsort(cl_vector)[-n_words:]
            row = []
            for k in highest:
                word = ngram_vectorizer._inverse_token_dictionary_[k]
                row.append(word)
            # w2 = ngram_vectorizer._inverse_token_dictionary_[second_]
            row = np.array(row)
            cluster_keywords.append(row)
        keywords.append(cluster_keywords)
        t_attrs = nx.get_node_attributes(TG.G, "slice_no")
    cl_attrs = nx.get_node_attributes(TG.G, "cluster_no")
    label_attrs = {}
    for node in TG.G.nodes():
        t_idx = t_attrs[node]
        cl_idx = cl_attrs[node]
        words = keywords[t_idx][cl_idx]
        s = ""
        for word in words[:-1]:
            s += word + sep
        s += word[-1]
        label_attrs[node] = s

    print("Complete.        ")
    nx.set_node_attributes(TG.G, label_attrs, "label")
    return TG


def time_semantic_plot(
    TG,
    semantic_axis,
    ax=None,
    vertices=None,
    label_edges=False,
    bundle=False,
    edge_scaling=1,
    node_kwargs={},
    edge_kwargs={},
):
    """
    Create a time-semantic plot of the graph ``TemporalGraph.G``.

    Parameters:
        TemporalGraph: temporal_mapper.TemporalGraph
            The temporal graph object to plot.
        semantic_axis: ndarray
            Array of shape ``(n_samples,)`` with the 1D semantic data to use in the plot.
        ax: matplotlib.axes (optional, default=None)
            Matplotlib axis to draw on
        vertices: list (optional, default=None)
            List of nodes in TG.G to include in the plot.
        label_edges: bool (optional, default=False)
            If true, include text labels of the edge weight on top of edges.
        edge_scaling: float (optional, default = 1)
            Scales the thickness of edges, larger is thicker.
        bundle: bool (optional, default=True)
            If true, bundle the edges of the graph using datashader's hammer_bundle function.
    Returns: matplotlib.axes

    """
    if ax is None:
        ax = plt.gca()
    if vertices is None:
        vertices = TG.G.nodes()
    G = TG.G.subgraph(vertices)

    pos = {}
    slice_no = nx.get_node_attributes(TG.G, "slice_no")
    semantic_axis = np.squeeze(semantic_axis)
    for node in vertices:
        t = slice_no[node]
        pt_idx = TG.get_vertex_data(node)
        w = TG.weights[t, pt_idx]
        node_ypos = np.average(semantic_axis[pt_idx], weights=w)
        node_xpos = np.average(TG.time[pt_idx], weights=w)
        pos[node] = (node_xpos, node_ypos)

    """ Plot nodes of graph. """
    node_size = [5 * np.log2(np.size(TG.get_vertex_data(node))) for node in vertices]
    if TG.n_components != 2:
        cval_dict = nx.get_node_attributes(TG.G, "cluster_no")
        node_clr = node_clr = [cval_dict[node] for node in vertices]
    else:
        clr_dict = nx.get_node_attributes(TG.G, "colour")
        node_clr = [clr_dict[node] for node in vertices]
    if bundle:
        alpha = 0.8
    else:
        alpha = 0.4
    if "alpha" in node_kwargs.keys():
        alpha = node_kwargs.pop("alpha")
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_clr,
        alpha=alpha,
        **node_kwargs,
    )
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
    ax.set_xticks(TG.checkpoints)
    ax.tick_params(axis="x", labelrotation=90)

    """ Plot edges of graph. """
    c = "k"
    if "c" in edge_kwargs.keys():
        c = edge_kwargs.pop("c")
    if "color" in edge_kwargs.keys():
        c = edge_kwargs.pop("color")
    if bundle == True:
        bundles = write_edge_bundling_datashader(TG, pos)
        x = bundles["x"].to_numpy()
        y = bundles["y"].to_numpy()
        ax.plot(x, y, c=c, lw=0.5 * edge_scaling, **edge_kwargs)
        if label_edges:
            print(
                "Warning: edge labels are not supported with bundling, consider passing bundle=False"
            )
    else:
        edge_width = np.array([np.log(d["weight"]) for (u, v, d) in G.edges(data=True)])
        edge_width /= np.amax(edge_width)
        elarge = [(u, v) for (u, v, d) in G.edges(data=True)]
        if "arrows" in edge_kwargs:
            arrows = edge_kwargs.pop("arrows")
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=elarge,
            width=edge_scaling * 2.5 * edge_width,
            arrows=False,
            edge_color=c,
            **edge_kwargs,
        )
        if label_edges:
            edge_labels = nx.get_edge_attributes(G, "weight")
            nx.draw_networkx_edge_labels(G, pos, edge_labels)

    return ax


def hex_desaturate(c, pc):
    """Desaturate c by pc%"""
    r, g, b, a = to_rgba(c)
    h, s, v = rgb_to_hsv((r, g, b))
    s *= pc
    r, g, b = hsv_to_rgb((h, s, v))
    return np.array([r, g, b, a])


def centroid_datamap(
    TG,
    ax=None,
    label_edges=False,
    vertices=None,
    edge_scaling=1,
    node_colouring="desaturate",
    bundle=True,
    node_kwargs={},
    edge_kwargs={},
):
    """Plot the temporal graph in 2d with vertices at their cluster centroids.

    Parameters:
        TemporalGraph: temporal_mapper.TemporalGraph
            The temporal graph object to plot.
        ax: matplotlib.axes (optional, default=None)
            Matplotlib axis to draw on
        node_colouring: ``'desaturate'`` or ``'override'`` (optional, default='desaturate')
            Determines how to incorporate temporal information in the color.
            The desaturate option will take the semantic colouring from datamapplot and desaturate points that are further back in time.
            The override option will throw away the semantic colouring and colour points only based on their time value.
        vertices: list (optional, default=None)
            List of nodes in TG.G to include in the plot.
        label_edges: bool (optional, default=False)
            If true, include text labels of the edge weight on top of edges.
        edge_scaling: float (optional, default = 1)
            Scales the thickness of edges, larger is thicker.
        bundle: bool (optional, default=True)
            If true, bundle the edges of the graph using datashader's hammer_bundle function.
    Returns: matplotlib.axes

    """
    if vertices is None:
        vertices = TG.G.nodes()
    G = TG.G.subgraph(vertices)
    if ax is None:
        ax = plt.gca()
    try:
        pos = nx.get_node_attributes(TG.G, "centroid")
    except AttributeError:
        TG.populate_node_attrs()
        pos = nx.get_node_attributes(TG.G, "centroid")

    """ Plot nodes of graph """
    node_size = [5 * np.log2(np.size(TG.get_vertex_data(node))) for node in vertices]
    slice_no = nx.get_node_attributes(TG.G, "slice_no")
    if node_colouring == "override":
        # Override cluster semantic colouring with time information
        node_clr = [slice_no[node] for node in vertices]
    elif node_colouring == "desaturate":
        # Keep semantic colouring and desaturate nodes in the past
        colour_dict = nx.get_node_attributes(TG.G, "colour")
        pc = [(slice_no[node] + 1) / TG.N_checkpoints for node in vertices]
        node_clr = [
            hex_desaturate(colour_dict[node], pc[i])
            for i, node in enumerate(colour_dict.keys())
        ]
    else:
        print("Accepted values of node_colouring are 'desaturate' and 'override'.")

    if bundle:
        alpha = 0.8
    else:
        alpha = 0.4
    if "alpha" in node_kwargs.keys():
        alpha = node_kwargs.pop("alpha")
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_clr,
        alpha=alpha,
        **node_kwargs,
    )

    """ Plot edges of graph """
    c = "k"
    if "c" in edge_kwargs.keys():
        c = edge_kwargs.pop("c")
    if "color" in edge_kwargs.keys():
        c = edge_kwargs.pop("color")
    if bundle == True:
        bundles = write_edge_bundling_datashader(TG, pos)
        x = bundles["x"].to_numpy()
        y = bundles["y"].to_numpy()

        ax.plot(x, y, c=c, lw=0.5 * edge_scaling, **edge_kwargs)
    else:
        edge_width = np.array([np.log(d["weight"]) for (u, v, d) in G.edges(data=True)])
        edge_width /= np.amax(edge_width)
        elarge = [(u, v) for (u, v, d) in G.edges(data=True)]
        if "arrows" in edge_kwargs:
            arrows = edge_kwargs.pop("arrows")
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=elarge,
            width=edge_scaling * 2.5 * edge_width,
            arrows=False,
            node_size=node_size,
            edge_color=c,
            **edge_kwargs,
        )
        if label_edges:
            tmp_dict = nx.get_edge_attributes(TG.G, "weight")
            edge_labels = {k: "{:.2f}".format(tmp_dict[k]) for k in tmp_dict}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)

    return ax


def export_to_javascript(path, TM):
    """write the javascript file for Roberta's edge bundling code."""
    try:
        pos = nx.get_node_attributes(TM.G, "centroid")
    except AttributeError:
        TM.populate_node_attrs()
        pos = nx.get_node_attributes(TM.G, "centroid")
    node_indices = {node: i for i, node in enumerate(pos.keys())}
    file = "const sampleData = {\n\tnodes: [\n"
    for node in TM.G.nodes():
        x, y = pos[node]
        file += "\t{" + f"x: {x}, y:{y}" + "},\n"
    file += "],\n edges: [\n"
    for src, dst, data in TM.G.edges(data=True):
        w = data["weight"]
        file += (
            "\t{"
            + f"source_node_idx: {node_indices[src]}, target_node_idx: {node_indices[dst]}"
            + "},\n"
        )
    file += "]\n}"
    with open(path, "w") as f:
        f.write(file)
        f.close()
    return file


def write_edge_bundling_datashader(TG, pos):
    """Use datashader to bundle edges from connected components together."""
    bundled_df = None
    for cpt in nx.connected_components(TG.G.to_undirected()):
        if len(cpt) == 1:
            continue
        cpt_subgraph = TG.G.subgraph(cpt)
        edge_df = DataFrame()
        node_df = DataFrame()
        cpt_pos = {node: pos[node] for node in cpt}
        node_idx = {node: i for i, node in enumerate(cpt_pos.keys())}
        node_df["name"] = cpt_pos.keys()
        node_df["x"] = [val[0] for val in cpt_pos.values()]
        node_df["y"] = [val[1] for val in cpt_pos.values()]
        edge_df["source"] = [node_idx[src] for src, dst in cpt_subgraph.edges()]
        edge_df["target"] = [node_idx[dst] for src, dst in cpt_subgraph.edges()]
        try:
            cpt_bundled_edges = hammer_bundle(node_df, edge_df)
        except ValueError:
            print(node_df)
            print(edge_df)
            print(cpt)
        if bundled_df is None:
            bundled_df = cpt_bundled_edges
        else:
            try:
                bundled_df = concat([bundled_df, cpt_bundled_edges])
            except ValueError:
                print(bundled_df)
                print(cpt_bundled_edges)
    return bundled_df


def sliceograph(TM, ax=None, clrs=["r", "g", "b"]):
    """Produce a sliceograph of a TemporalMapper

    Parameters:
        TemporalMapper: temporalmapper.TemporalMapper
            The temporal mapper object to plot.
        ax: matplotlib.axes (optional, default=None)
            Matplotlib axis to draw on
        clrs: list(str) (optional, default=['r','g','b'])
            A list of matplotlib colours, which will be cyclically to
            colour the intervals in the graph.

    Returns: matplotlib.axes

    """
    if ax is None:
        ax = plt.gca()
    ax.set_ylim(0, 1)
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
    for i in range(TM.N_checkpoints):
        offset = (0.01) * (i % 2) + 0.45
        slice_ = (TM.weights[i] >= 0.1).nonzero()[0]
        slice_max = max(TM.time[slice_])
        slice_min = min(TM.time[slice_])
        ax.plot([slice_min, slice_max], [offset, offset], c=clrs[i % len(clrs)])
    return ax
