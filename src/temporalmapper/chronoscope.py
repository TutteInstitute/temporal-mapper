"""UNCLASSIFIED // OFFICIAL USE ONLY / NON CLASSIFIÉ//RÉSERVÉ À DES FINS OFFICIELLES"""
from temporalmapper import TemporalMapper
import temporalmapper.kernels as tmwc
import temporalmapper.plotting as tmutils
import networkx as nx
import numpy as np
from tqdm import tqdm, trange
from copy import deepcopy 
from numpy import typing as npt
import json
import matplotlib.pyplot as plt

try:
    from toponymy import Toponymy
except ImportError as e:
    print(f"Chronoscope requires Toponymy: {e}")

class Chronoscope:
    SERIAL_VERSION = 1
    TOPONYMY_VERSION = 1
    """ Toponymy Integration """
    def __init__(
        self,
        time: npt.NDArray,
        data: npt.NDArray,
        text: list[str],
        embedding_vectors: npt.NDArray,
        mapper_params: dict,
        toponymy_params: dict,
        toponymy_fit_params: dict=None,
        verbose: bool=False,
    ):
        if np.size(time) != np.shape(data)[0]:
            raise AttributeError(
                "Number of datapoints",
                np.shape(data)[0],
                "does not equal number of timestamps",
                np.size(time),
            )
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        self.n_components = data.shape[1]
        self.time = time
        self.data = data
        self.text = text
        mapper_params['verbose'] = verbose
        self.mapper_params = mapper_params
        toponymy_params['verbose'] = verbose
        self.toponymy_params = toponymy_params
        self.n_layers = toponymy_params['clusterer'].max_layers
        self.embedding_vectors = embedding_vectors
        
        self.base_mapper = None
        self.mappers = {}
        self.toponymies = {}
        self.clusterers = {}
        self.slices = None
        self.clusters = None
        self.verbose = verbose
        self.toponymy_fit_params = toponymy_fit_params
        self.checkpoints = None
        self.title = "Chronoscope"
    
    def initialize_mapper(self):
        """ Create an initial Temporal Mapper to compute density and slices."""
        self.base_mapper = TemporalMapper(
            self.time,
            self.data,
            None, # no clusterer, as this will be handled by Toponymy
            **self.mapper_params,
        )
        self.base_mapper._compute_checkpoints()
        self.base_mapper._compute_density()
        self.base_mapper._compute_kernel_width()
        self.checkpoints = self.base_mapper.checkpoints
        self.density = self.base_mapper.density

    def slice(self):
        if self.base_mapper is None:
            self.initialize_mapper()
        kernel = self.mapper_params['kernel']
        if kernel == tmwc.square:
            eps = 0.01
        checkpoints = self.base_mapper.checkpoints
        data = self.data
        time = self.base_mapper.time
        densities = self.base_mapper.density
        weights = np.zeros((np.size(checkpoints), np.size(time)))
        cp_with_ends = [np.amin(time)] + list(checkpoints) + [np.amax(time)]
        slices = []
        for idx, t0 in tqdm(
            enumerate(checkpoints),
            disable = not self.verbose,
            desc = "Computing temporal slices",
            total = len(checkpoints)
        ):
            # get data in a given bin
            bin_width = (cp_with_ends[idx + 2] - cp_with_ends[idx]) / 2
            bin_width *= 1 / (2 -  self.base_mapper.g)
            if self.base_mapper.kernel_params == None:
                for i in np.arange(np.size(time)):
                    weights[idx, i] = kernel(
                        t0,
                        time[i],
                        densities[i],
                        bin_width,
                    )
            else:
                for i in np.arange(np.size(time)):
                    weights[idx, i] = kernel(
                        t0,
                        time[i],
                        densities[i],
                        bin_width,
                        params=self.base_mapper.kernel_params,
                    )
            slice_ = (weights[idx] >= eps).nonzero()
            slice_ = np.squeeze(slice_)
            slices.append(slice_)
        self.base_mapper.weights = weights
        self.slices = slices

    def cluster(self):
        """ Run Toponymy's clusterer at each time slice. """
        if self.base_mapper is None:
            self.initialize_mapper()
        if self.slices is None:
            self.slice()
        kernel = self.mapper_params['kernel']
        if kernel == tmwc.square:
            eps = 0.01
        # -2 as a placeholder for "not clustered"
        # magic number 4 = number of layers (todo make this a var)
        checkpoints = self.base_mapper.checkpoints
        data = self.data
        time = self.base_mapper.time
        densities = self.base_mapper.density
        clusters = np.ones((
            self.n_layers,
            np.size(checkpoints),
            np.size(time)),dtype=int
                          ) * -2
        cp_with_ends = [np.amin(time)] + list(checkpoints) + [np.amax(time)]
        cluster_layers = {}
        slices = []
        for idx, t0 in tqdm(
            enumerate(checkpoints),
            disable = not self.verbose,
            desc = "Toponymy clustering each time slice",
            total = len(checkpoints)
        ):
            slice_ = self.slices[idx]
            clusterer = deepcopy(self.toponymy_params['clusterer'])
            self.clusterers[idx] = clusterer
            clusterer.fit(
                clusterable_vectors=np.vstack(data[slice_]),
                embedding_vectors=self.embedding_vectors[slice_]
            )
            cluster_layers[idx] = clusterer.cluster_layers_
            n_layers = len(clusterer.cluster_layers_)
            if n_layers != self.n_layers:
                print(f"Warning! Slice {idx} has only {n_layers} out of {self.n_layers}!")
            for i in range(n_layers):
                clusters[i, idx, slice_] = clusterer.cluster_layers_[i].cluster_labels

        self.clusters = clusters
        weights = self.base_mapper.weights
        self.base_mapper.slices = slices
        self.cluster_layers = cluster_layers
        return (clusters, weights)       

    def compute_graph(self, layer):
        """ copy the base_mapper, manually assign clusters, and finish running it's graph construction."""
        TM = deepcopy(self.base_mapper)
        TM.clusters = self.clusters[layer,:,:]
        TM.add_vertices()
        TM.build_adj_matrix()
        TM.add_edges()
        TM.populate_edge_attrs()
        TM.populate_node_attrs()
        self.mappers[layer]=TM
        return TM

    def compute_toponymies(self):
        """ generate topic names at each time slice """
        if self.clusters is None:
            self.cluster()
        if self.verbose:
            print("Generating Toponymies. This may take a while...")
        kernel = self.mapper_params['kernel']
        if kernel == tmwc.square:
            eps = 0.01
        toponymies = {}
        checkpoints = self.base_mapper.checkpoints
        time = self.base_mapper.time
        data = self.data
        densities = self.base_mapper.density
        cp_with_ends = [np.amin(time)] + list(checkpoints) + [np.amax(time)]
        for idx, t0 in enumerate(self.base_mapper.checkpoints):
            slice_ = self.slices[idx]
            toponymy_params = self.toponymy_params
            toponymy_params['clusterer']=self.clusterers[idx]
            topic_model = Toponymy(**toponymy_params)
            topic_model.show_progress_bars
            topic_model.clusterer = self.clusterers[idx]
            topic_model.cluster_layers_ = topic_model.clusterer.cluster_layers_
            if idx > 0:
                for j, layer in enumerate(topic_model.cluster_layers_):
                   layer.previous_names = self.build_previous_names(idx, j)
            topic_model.fit(
                self.text[slice_],
                self.embedding_vectors[slice_],
                data[slice_],
                **self.toponymy_fit_params
            )
            toponymies[idx]=topic_model
            self.toponymies = toponymies
        return toponymies

    def build_previous_names(self, i, j):
        """ Find previous names for all clusters in toponomy i at layer j """
        mapper = self.mappers[j]
        previous_names = []
        n_unique = len(np.unique(mapper.clusters[i]))
        for c_idx in tqdm(
            range(n_unique),
            desc="Finding previous topic names",
        ):
            c = np.unique(mapper.clusters[i])[c_idx]
            if (c == -2) or (c == -1):
                continue
            vertex_name = str(i)+":"+str(c)
            previous_vertices = np.unique(mapper.get_dir_subvertices(vertex_name, backwards=True))
            ## returns list of "toponomy_number:cluster_number" strings
            tuples = [vertex.split(":") for vertex in previous_vertices]
            tuples = [t for t in tuples if int(t[0]) < i]
            names = [
                self.toponymies[int(time)].cluster_layers_[j].topic_names[int(cluster)]
                for time, cluster in tuples
            ]
            previous_names.append(names)
        return previous_names
        
    def connect_topics_to_nodes(self):
        """ create a dictionary mapping between topic names and dbmapper vertex names """
        topic_name_to_vertex = {}
        for i in range(self.n_layers):
            mapper = self.mappers[i]
            topic_names = {}
            for idx, t0 in enumerate(self.base_mapper.checkpoints):
                clusters = np.unique(mapper.clusters[idx])
                topic_model = self.toponymies[idx]
                cluster_layer = topic_model.cluster_layers_[i]
                for c in clusters:
                    topic_name = cluster_layer.topic_names[c]
                    vertex_name = str(i)+":"+str(idx)+":"+str(c)
                    topic_names[str(idx)+":"+str(c)] = topic_name
                    topic_name_to_vertex[topic_name] = vertex_name
             
            nx.set_node_attributes(mapper.G, topic_names, "topic_name")
        self.topic_name_to_vertex = topic_name_to_vertex
        
    def fit(self):
        self.initialize_mapper()
        self.slice()
        self.cluster()
        for i in range(self.n_layers):
            self.compute_graph(i)
        self.compute_toponymies()
        self.connect_topics_to_nodes()

    def temporal_plot(
        self,
        layer,
        ax = None,
        cluster_labels: dict = None,
        cluster_label_kwargs: dict = None,
        vertices: list=None,
        edge_scaling: float = 1.0,
        node_scaling: float = 1.0,
        node_size_bounds: tuple[float] = (5,50),
        edge_weight_bounds: float = 0.1,
        node_size_scale: str ='sigmoid',
        node_kwargs: dict = {},
        edge_kwargs: dict = {},
        layout_optimization: str = "barycenter",
        layout_optimization_kwargs: dict = {},
        hide_deg2_labels = False,
    ):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(12,8))
            
        mapper = self.mappers[layer]
        if cluster_labels is None:
            cluster_labels = nx.get_node_attributes(mapper.G, "topic_name")
            cluster_labels = {
                k:tmutils.squarify_text(s) for k,s in cluster_labels.items()
            }
        if hide_deg2_labels:
            def label(s,d):
                if d==2:
                    return ''
                else:
                    return s
            cluster_labels = {
                node:label(s,mapper.G.degree(node)) for node,s in cluster_labels.items()
            }
        return mapper.temporal_plot(
            ax = ax,
            vertices = vertices,
            cluster_labels = cluster_labels,
            cluster_label_kwargs = cluster_label_kwargs,
            edge_scaling = edge_scaling,
            edge_weight_bounds = edge_weight_bounds,
            node_scaling = node_scaling,
            node_size_bounds = node_size_bounds,
            node_size_scale = node_size_scale,
            node_kwargs = node_kwargs, 
            edge_kwargs = edge_kwargs,
            layout_optimization = layout_optimization,
            layout_optimization_kwargs = layout_optimization_kwargs,
        )
    
    def _serialize_state(self) -> dict:
        return {
            "version": self.SERIAL_VERSION,
            "toponymy_version": self.TOPONYMY_VERSION,
            "title": self.title,
    
            # raw inputs
            "time": self.time,
            "data": self.data,
            "text": self.text,
            "embedding_vectors": self.embedding_vectors,
    
            # configs
            "mapper_params": self.mapper_params,
            "toponymy_params": self.toponymy_params,
            "toponymy_fit_params": self.toponymy_fit_params,
            "verbose": self.verbose,
    
            # derived state
            "checkpoints": self.base_mapper.checkpoints,
            "density": self.base_mapper.density,
            "weights": self.base_mapper.weights,
            "slices": self.slices,
            "clusters": self.clusters,
    
            # graph metadata only
            "topic_names": {
                layer: nx.get_node_attributes(self.mappers[layer].G, "topic_name")
                for layer in self.mappers
            },
        }

    def save(self, path: str):
        import os, pickle
        
        os.makedirs(path, exist_ok=True)
        
        # 1. save lightweight state
        with open(os.path.join(path, "state.pkl"), "wb") as f:
            pickle.dump(self._serialize_state(), f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 2. save Toponymy objects separately
        for idx, topo in self.toponymies.items():
            fp = os.path.join(path, f"toponymy_{idx}.pkl")
            with open(fp, "wb") as f:
                pickle.dump(topo, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_toponymy(self, idx):
        if idx not in self.toponymies:
            if self.verbose:
                print(f"Recomputing Toponymy for slice {idx}")
            self.compute_toponymies()
        return self.toponymies[idx]


    @classmethod
    def load(cls, path: str, *, strict_toponymy: bool = True):
        import os, pickle
    
        # 1. load state
        with open(os.path.join(path, "state.pkl"), "rb") as f:
            state = pickle.load(f)
    
        version = state.get("version", 0)
        if version > cls.SERIAL_VERSION:
            raise RuntimeError(
                f"Chronoscope state v{version} > code v{cls.SERIAL_VERSION}"
            )
    
        #state = cls._migrate_state(state, version)
    
        # 2. reconstruct core object
        scope = cls(
            time=state["time"],
            data=state["data"],
            text=state["text"],
            embedding_vectors=state["embedding_vectors"],
            mapper_params=state["mapper_params"],
            toponymy_params=state["toponymy_params"],
            toponymy_fit_params=state["toponymy_fit_params"],
            verbose=state["verbose"],
        )
    
        scope.title = state["title"]
        scope.slices = state["slices"]
        scope.clusters = state["clusters"]
    
        # 3. rebuild base mapper
        scope.initialize_mapper()
        scope.base_mapper.checkpoints = state["checkpoints"]
        scope.base_mapper.density = state["density"]
        scope.base_mapper.weights = state["weights"]
    
        # 4. rebuild graphs
        for layer in range(scope.n_layers):
            TM = scope.compute_graph(layer)
            nx.set_node_attributes(
                TM.G,
                state["topic_names"].get(layer, {}),
                "topic_name",
            )
    
        # 5. load Toponymy objects
        scope.toponymies = {}
        for fname in os.listdir(path):
            if fname.startswith("toponymy_") and fname.endswith(".pkl"):
                idx = int(fname.split("_")[1].split(".")[0])
                with open(os.path.join(path, fname), "rb") as f:
                    topo = pickle.load(f)
    
                # optional compatibility check
                if strict_toponymy and hasattr(topo, "VERSION"):
                    if topo.VERSION != state["toponymy_version"]:
                        raise RuntimeError(
                            f"Toponymy version mismatch at slice {idx}"
                        )
    
                scope.toponymies[idx] = topo
    
        return scope

    def treemap(self):
        return tmutils.treemap