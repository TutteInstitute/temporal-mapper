import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from vectorizers.transformers import InformationWeightTransformer
from vectorizers import NgramVectorizer
from tqdm import tqdm, trange
from matplotlib.colors import to_rgba, rgb_to_hsv, hsv_to_rgb
from datashader.bundling import hammer_bundle
from pandas import DataFrame, concat
import plotly.graph_objects as go
import io, contextlib

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
    for i in trange(len(TG.slices), desc='Generating keywords'):
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

    nx.set_node_attributes(TG.G, label_attrs, "label")
    return label_attrs


def compute_time_semantic_positions(
    TG,
    semantic_axis,
    layout_optimization='barycenter',
    layout_optimization_kwargs = {},
):
    """ Compute node positions """
    x_pos = {}
    y_pos = {}
    slice_no = nx.get_node_attributes(TG.G, "slice_no")
    semantic_axis = np.squeeze(semantic_axis)
    for node in TG.G.nodes():
        t = slice_no[node]
        pt_idx = TG.get_vertex_data(node)
        w = TG.weights[t, pt_idx]
        y_pos[node] = np.average(semantic_axis[pt_idx], weights=w)
        x_pos[node] = np.average(TG.time[pt_idx], weights=w)
        
    if layout_optimization == "force-directed":
        y_init = [y_pos[node] for node in TG.G.nodes()]
        y_pos = force_directed_y_layout(TG.G, x_pos, y_init=y_init, **layout_optimization_kwargs)
    if layout_optimization == "barycenter":
        y_pos = temporal_barycenter_layout(TG.G, x_pos, **layout_optimization_kwargs) 
        
    pos = {node: (x_pos[node], y_pos[node]) for node in TG.G.nodes()}
    nx.set_node_attributes(TG.G, pos, name="ts_pos")


def plot_text_labels(
    axis,
    vertices,
    vertex_positions,
    vertex_labels,
    vertex_label_kwargs,
):
    texts = []
    from adjustText import adjust_text
    for node in vertices:
        x,y = vertex_positions[node]
        texts.append(
            axis.text(x, y, vertex_labels.get(node,''), **vertex_label_kwargs)
        )
    with contextlib.redirect_stdout(io.StringIO()):
        # For some reason, adjust_text keeps printing stuff
        texts, patches = adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-",color='k', alpha=0.25),
            ax=axis,
            min_arrow_len=1,
            avoid_self=False,
            expand_axes=True,
            time_lim = 5,
        )
    return axis

def time_semantic_plot(
    TG,
    semantic_axis,
    ax=None,
    vertices=None,
    cluster_labels={},
    cluster_label_kwargs={},
    edge_labels=None,
    bundle=False,
    layout_optimization='barycenter',
    edge_scaling=1,
    node_scaling=1,
    node_size_bounds: tuple[float] = (5,25),
    edge_weight_bounds: tuple[float] = (5,25),
    node_size_scale='linear',
    node_kwargs={},
    edge_kwargs={},
):
    """
    Create a time-semantic plot of the graph ``TemporalGraph.G``.

        Parameters
        ----------
        TemporalGraph: temporal_mapper.TemporalGraph
            The temporal graph object to plot.
        semantic_axis: ndarray
            Array of shape ``(n_samples,)`` with the 1D semantic data to use in the plot.
        ax: matplotlib.axes (optional, default=None)
            Matplotlib axis to draw on
        vertices: list (optional, default=None)
            List of nodes in TG.G to include in the plot.
        cluster_labels: dict (optional, default={})
            Dictionary of labels with `cluster_labels[node]` a string to label vertex `node`.
        cluster_label_kwargs: dict (optional, default={})
            Keyword arguments for `matplotlib.axis.text` used when plotting cluster labels.
        edge_labels: dict (optional, default=None)
            Dictionary of labels with `edge_labels[e]` a string to label edge `e`.
        bundle: bool (optional, default=False)
            If true, uses the edge-bundling algorithm from datashader to plot edges.
        layout_optimization: string (optional, default='barycenter')
            Optimization method used to reduce edge-crossings: one of None, "none", "force-directed" or "barycenter"
        edge_scaling: float (optional, default = 1)
            Scales the thickness of edges, larger is thicker.
        node_scaling: float (optional, default = 10)
            Scales the size of vertices
        node_size_scale: string (optional, default='linear')
            Specifies linear or logarithmic scaling for node sizes
        bundle: bool (optional, default=False)
            If true, bundle the edges of the graph using datashader's hammer_bundle function.
        node_kwargs: dict (optional, default={})
            Keyword arguments passed to networkx.draw_networkx_nodes()
        edge_kwargs: dict (optional, default={})
            Keyword arguments passed to networkx.draw_networkx_edges()
            
        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the temporal plot.

    """
    if ax is None:
        ax = plt.gca()
    if vertices is None:
        vertices = TG.G.nodes()
    G = TG.G.subgraph(vertices)
    compute_time_semantic_positions(TG, semantic_axis, layout_optimization = layout_optimization)
    pos = nx.get_node_attributes(TG.G,'ts_pos')
    """ Plot nodes of graph. """
    node_size = compute_node_size(
        TG,
        G,
        node_scaling,
        node_size_scale,
        node_size_bounds
    )
        
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
    if bundle == True:
        bundles = write_edge_bundling_datashader(TG, pos)
        x = bundles["x"].to_numpy()
        y = bundles["y"].to_numpy()
        ax.plot(x, y, lw=0.5 * edge_scaling, **edge_kwargs)
        if edge_labels is not None:
            print(
                "Warning: edge labels are not supported with bundling, consider passing bundle=False"
            )
    else:
        edge_width = np.array([np.log(d["weight"]) for (u, v, d) in G.edges(data=True)])
        if len(edge_width)>0:
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
            **edge_kwargs,
        )
        if edge_labels is not None:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
    if len(cluster_labels)>0:
        ax = plot_text_labels(
            axis = ax,
            vertices = vertices,
            vertex_positions = pos,
            vertex_labels = cluster_labels,
            vertex_label_kwargs = cluster_label_kwargs,
        )
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
    edge_labels=None,
    vertices=None,
    edge_scaling=1,
    node_colouring="desaturate",
    bundle=True,
    node_kwargs={},
    edge_kwargs={},
):
    """Plot the temporal graph in 2d with vertices at their cluster centroids.

        Parameters
        ----------
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
        edge_labels: dict (optional, default=None)
            Dictionary of labels with edge_labels[e] a string to label edge e.
        edge_scaling: float (optional, default = 1)
            Scales the thickness of edges, larger is thicker.
        bundle: bool (optional, default=True)
            If true, bundle the edges of the graph using datashader's hammer_bundle function.
        node_kwargs: dict (optional, default={})
            Keyword arguments passed to networkx.draw_networkx_nodes()
        edge_kwargs: dict (optional, default={})
            Keyword arguments passed to networkx.draw_networkx_edges()
            
        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the centroid datamap.

    """
    if vertices is None:
        vertices = TG.G.nodes()
    G = TG.G.subgraph(vertices)
    if ax is None:
        ax = plt.gca()
    try:
        pos = nx.get_node_attributes(G, "centroid")
    except AttributeError:
        TG.populate_node_attrs()
        pos = nx.get_node_attributes(G, "centroid")

    """ Plot nodes of graph """
    node_size = np.array([5 * np.log2(np.size(TG.get_vertex_data(node))) for node in vertices])
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
            for i, node in enumerate(vertices)
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
        bundles = write_edge_bundling_datashader(TG, pos, vertices=vertices)
        x = bundles["x"].to_numpy()
        y = bundles["y"].to_numpy()
        if len(edge_kwargs.keys()) > 0:
            print("Warning! You have passed edge_kwargs with bundle=True, which is not supported.")
        ax.plot(x, y, c=c, lw=0.5 * edge_scaling)
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
        if edge_labels is not None:
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


def write_edge_bundling_datashader(TG, pos, vertices=None):
    """Use datashader to bundle edges from connected components together."""
    if vertices is None:
        vertices = TG.G.nodes()
    G = TG.G.subgraph(vertices)
    bundled_df = None
    for cpt in nx.connected_components(G.to_undirected()):
        if len(cpt) == 1:
            continue
        cpt_subgraph = G.subgraph(cpt)
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

from scipy.optimize import minimize
def force_directed_y_layout(G, x_positions, y_init=None, iterations=1000, edge_weight=1.0, repulsion_weight=0.1):
    """
    Use force-directed algorithm to find y-positions that minimize crossings
    X-positions are fixed (time), optimize only y-positions
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize y-positions randomly
    if y_init is None:
        y_init = np.random.random(n)
    
    # Callback for progress tracking
    iteration_count = [0]
    pbar = tqdm(total=iterations, desc="Optimizing layout")
    
    def callback(xk):
        iteration_count[0] += 1
        pbar.update(1)
    
    def energy(y_positions):
        """
        Energy function to minimize:
        - Edge length (keep connected nodes close in y)
        - Node repulsion (spread nodes apart to avoid overlap)
        """
        energy = 0
        
        # Edge attraction: minimize vertical distance between connected nodes
        for u, v in G.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            y_diff = y_positions[i] - y_positions[j]
            x_diff = x_positions[u] - x_positions[v]
            # Penalize y-distance, weighted by x-distance
            energy += edge_weight * y_diff**2 / (abs(x_diff) + 0.1)
        
        # Node repulsion: keep nodes separated
        for i in range(n):
            for j in range(i+1, n):
                y_diff = y_positions[i] - y_positions[j]
                x_diff = x_positions[nodes[i]] - x_positions[nodes[j]]
                dist = np.sqrt(x_diff**2 + y_diff**2)
                if dist > 0:
                    energy -= repulsion_weight / dist
        
        return energy
    
    # Optimize with callback
    result = minimize(energy, y_init, method='L-BFGS-B', 
                     callback=callback,
                     options={'maxiter': iterations})
    
    pbar.close()
    
    y_positions = {node: result.x[i] for i, node in enumerate(nodes)}
    return y_positions

def temporal_barycenter_layout(
    G,
    x_positions,
    y_positions=None,
    iterations=1000,
    lr_init=0.8,
    lr_min=0.05,
    lr_max=1.0,
    momentum=0.8,
    tol=1e-4,
    decay=0.005,
    eps=1e-6,
):
    """
    Barycenter-based layout for edge-crossing minimization with:
    - adaptive learning rate
    - momentum
    - normalization
    - early stopping
    """

    nodes = list(G.nodes())
    edge_weights = nx.get_edge_attributes(G, "weight")

    # --- Initialize y positions ---
    if y_positions is None:
        y_positions = {n: np.random.uniform(-1, 1) for n in nodes}
    else:
        y_positions = dict(y_positions)

    # --- Velocity for momentum ---
    velocity = {n: 0.0 for n in nodes}
    prev_avg_delta = None

    # --- Sanity check ---
    if not set(nodes).issubset(x_positions):
        raise ValueError("x_positions must contain all nodes")

    for it in range(iterations):
        new_y = {}
        total_delta = 0.0

        # --- Base learning rate ---
        lr = lr_init * np.exp(-decay * it)
        lr = np.clip(lr, lr_min, lr_max)

        # --- Compute barycenter attraction ---
        for node in nodes:
            yi = y_positions[node]
            xi = x_positions[node]
        
            neighbors = list(G.neighbors(node))
            attraction = 0.0
        
            if neighbors:
                weighted_sum = 0.0
                weight_total = 0.0
        
                for nbr in neighbors:
                    dx = abs(xi - x_positions[nbr])
        
                    # --- Edge weight (default = 1.0 if missing) ---
                    ew = edge_weights.get((node, nbr),
                         edge_weights.get((nbr, node), 1.0))
        
                    # --- Combined weight: spatial + edge importance ---
                    w = ew / (dx + eps)
        
                    weighted_sum += w * y_positions[nbr]
                    weight_total += w
        
                target = weighted_sum / weight_total
                attraction = target - yi
        
            # --- Momentum update ---
            v = momentum * velocity[node] + lr * attraction
            velocity[node] = v
        
            new_y[node] = yi + v
            total_delta += abs(v)

        # --- Normalize to prevent drift ---
        vals = np.array(list(new_y.values()))
        std = vals.std()
        if std > 0:
            mean = vals.mean()
            new_y = {n: (y - mean) / std for n, y in new_y.items()}

        avg_delta = total_delta / len(nodes)

        # --- Adaptive LR correction ---
        if prev_avg_delta is not None:
            if avg_delta > prev_avg_delta:
                lr_init *= 0.7
            else:
                lr_init *= 1.05
            lr_init = np.clip(lr_init, lr_min, lr_max)

        y_positions = new_y
        prev_avg_delta = avg_delta

        # --- Early stopping ---
        if avg_delta < tol:
            break

    return y_positions

def compute_node_size(
    mapper,
    G,
    node_scaling,
    node_size_scale,
    node_size_bounds,
):
    smin,smax = node_size_bounds
    if node_size_scale == 'logarithmic':
        node_size = [node_scaling * np.log2(np.size(mapper.get_vertex_data(node))) for node in G.nodes()]
    elif node_size_scale == 'linear':
        node_size = np.array([np.size(mapper.get_vertex_data(node)) for node in G.nodes()], dtype=np.float64)
        node_size = node_size*node_scaling
    elif node_size_scale == 'sigmoid':
        raw = np.array(
            [node_scaling*np.size(mapper.get_vertex_data(node)) for node in G.nodes()],
            dtype=np.float64
        )
        mu = raw.mean()
        sigma = raw.std() if raw.std() > 0 else 1.0
        z = (raw - mu) / sigma
        sig = 1.0 / (1.0 + np.exp(-z))
        node_size = smin + (smax - smin) * sig
    else:
        raise ValueError("node_size_scale keyword argument must be 'linear' or 'logarithmic'.")
    node_size = [np.clip(s,smin,smax) for s in node_size]
    return node_size


def prepare_plotly_graph_objects(
    mapper,
    positions,
    hover_text = {},
    edge_scaling: float = 1,
    node_scaling: float = 1,
    node_size_bounds: tuple[float] = (5,25),
    edge_weight_bounds: tuple[float] = (5,25),
    node_size_scale: str = 'linear',
):
    # https://plotly.com/python/network-graphs/
    edge_traces = []
    G = mapper.G
    clr_dict = nx.get_node_attributes(G, "colour")
    weight = nx.get_edge_attributes(G, "weight")
    wmin, wmax = edge_weight_bounds
    edge_size_dict = {
        e:edge_scaling*np.clip(weight[e],wmin,wmax) for e in G.edges()
    }
    for (u, v) in G.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
    
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                hoverinfo="none",
                line=dict(
                    width=edge_size_dict[(u,v)],
                    color=clr_dict[u],
                )
            )
        )
        
    node_x = []
    node_y = []
    colours = []
    labels = []

    node_size = compute_node_size(
        mapper,
        G,
        node_scaling,
        node_size_scale,
        node_size_bounds
    )
    for node in G.nodes():
        x,y = positions[node]
        node_x.append(x)
        node_y.append(y)
        label_str = hover_text[node]
        labels.append(label_str)
        colours.append(G.nodes[node]['colour'])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            size=node_size,
            sizemode='area',
            color=colours
        ),
        text=labels
    )
    return edge_traces, node_trace
