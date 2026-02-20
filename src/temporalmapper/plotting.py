from warnings import warn
import io, contextlib

from tqdm import tqdm, trange
from matplotlib.colors import to_rgba, rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pandas import DataFrame, concat
from datashader.bundling import hammer_bundle
from sklearn.utils.validation import check_is_fitted

from vectorizers.transformers import InformationWeightTransformer
from vectorizers import NgramVectorizer

from temporalmapper.layout import (
    temporal_barycenter_layout,
    component_ordered_layout,
    force_directed_y_layout,
    compute_time_semantic_positions,
)
from temporalmapper.analytics import (
    nodes_in_slice,
)

def squarify_text(text):
    """Make a string more square by adding newlines"""
    words = text.split()
    if not words:
        return ""

    total_chars = sum(len(w) for w in words) + len(words) - 1
    target_width = np.ceil(np.sqrt(total_chars))

    lines = []
    current_line = []

    for word in words:
        # length if we add this word to the current line
        projected_len = sum(len(w) for w in current_line) + len(current_line) + len(word)

        if projected_len <= target_width:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)

def generate_keyword_labels(word_bags, TG, ngram_vectorizer=None, n_words=3, sep=" "):
    """Using a bag of words corresponding to each data point, get top n_words informative
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
    layout_optimization_kwargs={},
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
            Specifies linear, sigmoid or logarithmic scaling for node sizes
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
    if (layout_optimization == 'barycenter') and ('spacing' not in layout_optimization_kwargs.keys()):
        layout_optimization_kwargs['spacing']=np.sqrt(node_scaling)
    
    compute_time_semantic_positions(
        TG,
        semantic_axis,
        layout_optimization = layout_optimization,
        layout_optimization_kwargs=layout_optimization_kwargs,
    )
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

def slice_df(mapper, idx):
    G = mapper.G
    counts = nx.get_node_attributes(G, 'count')
    growth = nx.get_node_attributes(G, 'growth')
    
    nodes = nodes_in_slice(mapper, idx)
    top_n = 5

    count_values = np.array([
        counts[v] for v in nodes    
    ])
    growth_values = np.array([
        growth[v] for v in nodes
    ])
    slice_df = DataFrame({
        'node':nodes,
        'slice':[idx]*len(nodes),
        'count':count_values,
        'growth':growth_values,
    })
    return slice_df

def growth_map(mapper, index=None):
    try:
        import plotly.express as px
    except ImportError as e:
        warn("Interactive growth map requires plotly")
        raise e

    check_is_fitted(mapper, ["is_fitted_"])
    if index is None:
        dfs = []
        for index in range(mapper.N_checkpoints):
            dfs.append(slice_df(mapper, index))
        dataframe = concat(dfs, ignore_index=True) 
        path = ['slice', 'node']
    else:
        dataframe = slice_df(mapper, index)
        path = ['node']

    fig = px.treemap(
        dataframe,
        path=path,
        values='count',
        color='growth',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
    )
        

    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Growth: %{color:.2f}'
    )

    return fig

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


def compute_node_size(
    mapper,
    G,
    node_scaling,
    node_size_scale,
    node_size_bounds,
):
    smin,smax = node_size_bounds
    if node_size_scale == 'log':
        node_size_scale = 'logarithmic'
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
        raise ValueError("node_size_scale keyword argument must be 'linear' 'sigmoid' or 'logarithmic'.")
    node_size = [np.clip(s,smin,smax) for s in node_size]
    return node_size


def prepare_plotly_graph_objects(
    mapper,
    positions,
    hover_text = {},
    custom_data = {},
    edge_scaling: float = 1,
    node_scaling: float = 1,
    node_size_bounds: tuple[float] = (5,25),
    edge_weight_bounds: tuple[float] = (0.1,5),
    node_size_scale: str = 'linear',
):
    # https://plotly.com/python/network-graphs/
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        warn("Interactive plotting requires plotly")
        raise e

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
                    width=edge_size_dict.get((u,v),'0.5'),
                    color=clr_dict.get(u,'black'),
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
        text=labels,
        customdata=[custom_data[node] for node in G.nodes()],
    )
    return edge_traces, node_trace
