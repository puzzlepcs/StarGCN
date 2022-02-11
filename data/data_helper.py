import os, inspect, pickle
from time import time
import numpy as np
import scipy.sparse as sp

from src.utils import generate_incidence_matrix, generate_adjacency_matrix

def load_data(cfg):
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    d = os.path.join(current, cfg["data"])

    # Load hypergraph
    hypergraph = load_pickle(d, "hypergraph.pickle")
    hypergraph = {key: list(value) for key, value in hypergraph.items()}

    # Load node labels
    labels = load_pickle(d, "labels.pickle")

    m_hyperedge = len(hypergraph)
    n_node = len(labels)
    num_class = len(set(labels))

    # Load train / test split
    split = cfg["split"]
    l = f"splits_{cfg['labels']}" if cfg['labels'] != "" else "splits"
    s = load_pickle(os.path.join(d, l), f"{split}.pickle")
    train_idx = s["train"]
    test_idx = s["test"]

    # Load node_features
    node_features = load_pickle(d, "features.pickle").todense()
    input_dim = node_features.shape[1]

    # Make hyperedge features
    hyperedge_features = initiate_hyperedge_features(hypergraph, node_features)

    # Incidence Matrix
    try:
        inc_mat = sp.load_npz(os.path.join(d, "s_pre_inc_mat.npz"))
    except:
        inc_mat = generate_incidence_matrix(hypergraph, n_node, m_hyperedge)
        sp.save_npz(os.path.join(d, 's_pre_inc_mat.npz'), inc_mat)

    # Adjacency Matrix of Star expansion
    try:
        adj_mat = sp.load_npz(os.path.join(d, 's_pre_adj_mat.npz'))
    except:
        adj_mat = generate_adjacency_matrix(inc_mat, n_node, m_hyperedge)
        sp.save_npz(os.path.join(d, 's_pre_adj_mat.npz'), adj_mat)

    data = {
        'hypergraph': hypergraph,
        'features': np.concatenate([node_features, hyperedge_features], axis=0),
        'labels': labels,
        'n_node': n_node,
        'm_hyperedge': m_hyperedge,
        'num_class': num_class,
        'input_dim': input_dim,
        'inc_mat': inc_mat,
        'adj_mat': adj_mat,
        'train_idx': train_idx,
        'test_idx': test_idx
    }
    return data


def load_pickle(d, file_name):
    with open(os.path.join(d, file_name), "rb") as handle:
        file = pickle.load(handle)
    return file


def load_txt(d, file_name) -> dict:
    hypergraph = dict()
    nodes = set()
    h_idx = 0
    with open(os.path.join(d, file_name), "r") as file:
        for line in file:
            hyperedge = {int(node) for node in line.strip().split(", ")}
            h_idx += 1
            hypergraph[h_idx] = hyperedge
            nodes = nodes | hyperedge
    n_node = len(nodes)
    m_hyperedge = len(hypergraph)

    return hypergraph, n_node, m_hyperedge


def load_embeddings(d:str, file_name:str) -> dict:
    embs = {}
    with open(os.path.join(d, file_name), "r") as file:
        for i, line in enumerate(file):
            l = line.strip().split()
            embs[int(l[0])] = [float(e) for e in l[1:]]
    return embs


def initiate_hyperedge_features(hypergraph, node_features):
    # Create hyperedge features by
    # taking average of nodes inculded in the corresponding hyperedge
    hyperedge_features = []
    for key, value in hypergraph.items():
        hyperedge_feat = node_features[list(value)]
        hyperedge_feat = hyperedge_feat.mean(axis=0)
        hyperedge_features.append(hyperedge_feat)
    hyperedge_features = np.stack(hyperedge_features, axis=0)

    return hyperedge_features