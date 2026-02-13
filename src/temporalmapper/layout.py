import networkx as nx
from copy import deepcopy
from tqdm import tqdm
from sklearn.decomposition import PCA

def compute_time_semantic_positions(
    TG,
    semantic_axis,
    layout_optimization='ordered',
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
    if layout_optimization == "ordered":
        y_pos = component_ordered_layout(TG.G, x_pos, **layout_optimization_kwargs)
        
        
    pos = {node: (x_pos[node], y_pos[node]) for node in TG.G.nodes()}
    nx.set_node_attributes(TG.G, pos, name="ts_pos")

def construct_components(G):
    sources = [node for node in G.nodes() if G.in_degree(node)==0]
    components = {s:[] for s in sources}
    for node in G.nodes():
        shortest_source = node
        min_path = np.inf
        for s in sources:
            try:
                l = nx.shortest_path_length(G, source=s, target=node, weight='weight')
            except nx.NetworkXNoPath:
                l = np.inf
            if l < min_path:
                min_path = l
                shortest_source = s
        components[shortest_source].append(node)
    return components

def min_path_between_sets(G, cp1, cp2):
    G_quotient = deepcopy(G)
    
    cp1 = set(cp1)
    cp2 = set(cp2)
    
    # Ensure cp1 and cp2 don't overlap
    if cp1 & cp2:
        return 0
    
    # Create super-nodes for each set
    u = 'cp1_super'
    v = 'cp2_super'
    
    # Add super-nodes
    G_quotient.add_node(u)
    G_quotient.add_node(v)
    
    # Connect all vertices in cps to super-nodes with zero-weight edges
    for node in cp1:
        G_quotient.add_edge(u, node, weight=0)
    for node in cp2:
        G_quotient.add_edge(node, v, weight=0)

    try:
        path_length = nx.shortest_path_length(G_quotient, u, v, weight='weight')
        return path_length
    except nx.NetworkXNoPath:
        return np.inf      
        
import numpy as np
from collections import deque

def find_connected_components(distances):
    """
    Find groups of objects that are mutually reachable (finite distances).
    Returns list of lists, where each inner list contains indices of a connected component.
    """
    n = len(distances)
    visited = set()
    components = []
    
    for start in range(n):
        if start in visited:
            continue
        
        # BFS to find all reachable nodes from start
        component = []
        queue = deque([start])
        visited.add(start)
        
        while queue:
            node = queue.popleft()
            component.append(node)
            
            # Find all neighbors with finite distance
            for neighbor in range(n):
                if (neighbor not in visited and 
                    distances[node][neighbor] != np.inf):
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        components.append(sorted(component))
    
    return components


def nearest_neighbor_ordering(distances, start=0):
    """
    Greedy algorithm: always go to the nearest unvisited object.
    
    Args:
        distances: NxN numpy array where distances[i][j] is distance from i to j
        start: starting object index
    
    Returns:
        numpy array of indices representing the ordering
    """
    n = len(distances)
    unvisited = set(range(n))
    current = start
    path = [current]
    unvisited.remove(current)
    
    while unvisited:
        # Vectorized: get distances to all unvisited nodes
        unvisited_list = list(unvisited)
        distances_to_unvisited = distances[current][unvisited_list]
        
        # Filter out infinite distances
        finite_mask = distances_to_unvisited != np.inf
        if not np.any(finite_mask):
            break  # No more reachable nodes
        
        nearest_idx = unvisited_list[np.argmin(
            np.where(finite_mask, distances_to_unvisited, np.inf)
        )]
        path.append(nearest_idx)
        unvisited.remove(nearest_idx)
        current = nearest_idx
    
    return np.array(path)


def best_nearest_neighbor(distances, component):
    """
    Try nearest neighbor from each starting point within a component, 
    return the best.
    
    Args:
        distances: Full NxN distance matrix
        component: List of indices in this component
    
    Returns:
        (best_path, best_distance) for this component
    """
    best_path = None
    best_distance = np.inf
    
    for start in component:
        path = nearest_neighbor_ordering(distances, start)
        
        if len(path) == 0:
            continue
        
        total = np.sum(distances[path[:-1], path[1:]])
        
        if total < best_distance:
            best_distance = total
            best_path = path
    
    return best_path, best_distance


def two_opt_fast(distances, path, max_iterations=1000):
    """
    Faster 2-opt using numpy vectorization for distance calculations.
    """
    def path_distance(p):
        if len(p) <= 1:
            return 0
        return np.sum(distances[p[:-1], p[1:]])
    
    path = np.array(path, dtype=int)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        best_improvement = 0
        best_i, best_j = None, None
        
        # Calculate all improvements at once
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                # Change in distance from reversing path[i:j]
                # Only need to check the edges that change
                d1 = distances[path[i-1], path[i]] + distances[path[j-1], path[j]]
                d2 = distances[path[i-1], path[j-1]] + distances[path[i], path[j]]
                improvement = d1 - d2
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i, best_j = i, j
        
        if best_improvement > 0:
            path = np.concatenate([path[:best_i], path[best_i:best_j][::-1], path[best_j:]])
            improved = True
        
        iteration += 1
    
    return path


def arrange_components(G,cpts):
    n_cpts = len(cpts)
    distances = np.array(
        [[min_path_between_sets(G, cpts[cp1],cpts[cp2]) for cp2 in cpts] for cp1 in cpts]
    )
    # Find connected components
    components = find_connected_components(distances)
    #print(f"Found {len(components)} connected component(s): {components}")
    
    all_paths = []
    total_distance = 0
    
    # Optimize each component
    for component in components:
        #print(f"\nOptimizing component {component}...")
        
        if len(component) == 1:
            # Single object, no optimization needed
            path = np.array(component)
            all_paths.append(path)
            #print(f"  Single object: {path}")
        else:
            # Multi-start greedy
            path, greedy_dist = best_nearest_neighbor(distances, component)
            #print(f"  Greedy distance: {greedy_dist}")
            
            # Improve with 2-opt
            path = two_opt_fast(distances, path)
            component_distance = np.sum(distances[path[:-1], path[1:]])
            #print(f"  After 2-opt: {component_distance}")
            #print(f"  Path: {path}")
            
            all_paths.append(path)
            total_distance += component_distance
    
    # Concatenate all paths
    final_path = np.concatenate(all_paths)
    
    return final_path, total_distance, components


    if path is None:
        raise ValueError("No valid ordering found (some objects unreachable)")
    
    print(f"Greedy distance: {greedy_dist}")
    
    # Improve with 2-opt
    path = two_opt(distances, path)
    
    total_distance = np.sum(distances[path[:-1], path[1:]])
    
    return path, total_distance


def component_ordered_layout(G, x_positions, spacing=5):
    cpts = construct_components(G)
    order = np.concatenate(arrange_components(G, cpts)[2])
    y_pos = {}
    cpt_sources = list(cpts.keys())
    current_y_value = 0
    for i in order:
        s = cpt_sources[i]
        nodes = cpts[s]
        subgraph = G.subgraph(nodes).copy()
        cpt_y_pos = temporal_barycenter_layout(
            subgraph,
            x_positions,
            spacing=spacing,
        )
        cpt_min = np.min(np.array([
           cpt_y_pos[node] for node in nodes 
        ]))
        cpt_max = np.max(np.array([
            cpt_y_pos[node] for node in nodes
        ]))
        for node in nodes:
            y_pos[node] = cpt_y_pos[node]-cpt_min+current_y_value
        current_y_value += cpt_max-cpt_min + spacing
    return y_pos

def initial_y_positions(G):
    # Find minimal arborescence (directed spanning tree)
    arb = nx.minimum_spanning_arborescence(G)
    
    # Find root nodes (nodes with in_degree 0 in arborescence)
    roots = [node for node in arb.nodes() if arb.in_degree(node) == 0]
    
    x_pos = {}
    y_pos = {}
    
    def layout_tree(node, x, y_start, y_spacing):
        x_pos[node] = x
        y_pos[node] = y_start
        
        # Get children of this node in arborescence
        children = list(arb.successors(node))
        
        if children:
            # Fan out children vertically
            num_children = len(children)
            y_offset = (num_children - 1) * y_spacing / 2
            
            for i, child in enumerate(children):
                child_y = y_start - y_offset + i * y_spacing
                layout_tree(child, x + 1, child_y, y_spacing)
    
    # Layout each tree rooted at root nodes
    y_current = 0
    for root in roots:
        layout_tree(root, x=0, y_start=y_current, y_spacing=1)
        # Update y_current for next tree
        y_current = max(y_pos.values()) + 2
    
    return y_pos
    
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
    lambda_coparent=0.2,
    spacing=2,
):
    """
    Barycenter-based layout for edge-crossing minimization with:
    - adaptive learning rate
    - momentum
    - normalization
    - early stopping
    """

    def coparent_penalty(d, spacing=spacing):
        if d < spacing:
            return 0
        else:
            return d-spacing
    
    nodes = list(G.nodes())
    edge_weights = nx.get_edge_attributes(G, "weight")

    # --- Initialize y positions ---
    if y_positions is None:
        try:
            y_positions = initial_y_positions(G)
        except:
            # can happen if G is not connected
            y_positions = {n:np.random.random() for n in nodes}
    else:
        y_positions = dict(y_positions)

    # --- Velocity for momentum ---
    velocity = {n: 0.0 for n in nodes}
    coparent_force = {n: 0.0 for n in nodes}
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
            #neighbors = list(G.predecessors(node))
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

            # --- Co-parent attraction ---
            parents = list(G.predecessors(node))
            if len(parents) >= 2:
                mean_y = np.mean([y_positions[p] for p in parents])
                total_weight = np.sum([
                    edge_weights.get((p,node)) for p in parents
                ])
                for p in parents:
                    w = edge_weights.get((p, node))/total_weight
                    coparent_force[p] += w * lambda_coparent * (mean_y - y_positions[p])
        
            # --- Momentum update ---
            cp_force = coparent_penalty(coparent_force[node])
            v = momentum * velocity[node] + lr * (attraction + cp_force)
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