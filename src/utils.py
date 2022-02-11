import numpy as np
import os
import scipy.sparse as sp
import torch
from time import time


def normalize(mx):
    """
    Symmetrically normalize sparse matrix
    @param mx: input matrix
    @type mx: scipy sparse matrix
    @return: D^{-1/2} M D^{-1/2} (where D is the diagonal node-degree matrix)
    @rtype: scipy sparse mtrix
    """
    d = np.array(mx.sum(axis=1))  # D is the diagonal node-degree matrix

    d_hi = np.power(d, -0.5).flatten()
    d_hi[np.isinf(d_hi)] = 0.
    D_HI = sp.diags(d_hi)

    return D_HI.dot(mx).dot(D_HI)


def sparse_mx_to_torch_sparse_tensor(mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((mx.row, mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_incidence_matrix(hypergraph, n_node, m_hyperedge):
    print("generating incidence matrix...")
    start = time()
    h_idx, deg = 0, 0
    row, col = [], []
    for node_set in hypergraph.values():
        row.extend(list(node_set))
        col.extend([h_idx for _ in range(len(node_set))])
        h_idx += 1
        deg += len(node_set)
    inc_mat = sp.csr_matrix((np.ones(deg), (row, col)), shape=(n_node, m_hyperedge))
    end = time()

    print(f"costing {end - start:.4f}s, saved inc_mat...")
    return inc_mat


def generate_adjacency_matrix(inc_mat, n_node, m_hyperedge):
    I = inc_mat.tolil()

    print("generating adjacency matrix")
    start = time()

    # Make adjacency matrix using incidence matrix
    adj_mat = sp.dok_matrix(
        (n_node + m_hyperedge, n_node + m_hyperedge), dtype=np.float32).tolil()
    adj_mat[:n_node, n_node:] = I
    adj_mat[n_node:, :n_node] = I.T
    adj_mat = adj_mat.tocsr()

    end = time()
    print(f"costing {end - start:.4f}s, saved adj_mat...")
    return adj_mat


def generate_clique_adjacency_matrix(hypergraph, n_node):
    from itertools import combinations

    print("generating adjacency matrix of clique expansion...")
    start = time()

    edge_list = set()
    for hyperedge in hypergraph.values():
        pairwise = combinations(hyperedge, 2)
        for src, trg in pairwise:
            edge_list.add(tuple(sorted([src, trg])))
    sources = [edge[0] for edge in edge_list]
    targets = [edge[1] for edge in edge_list]
    adj_mat = sp.coo_matrix(
        (np.ones_like(sources), (sources, targets)),
        shape=(n_node, n_node), dtype=np.float32)
    adj_mat = adj_mat + adj_mat.T

    end = time()
    print(f"costing {end - start:.4f}s, saved adj_mat...")
    return adj_mat

def save_embeddings(save_dir, embeddings):
    ensure_dir(save_dir)

    with open(save_dir, "w") as f:
        for i, emb in enumerate(embeddings):
            l = [str(e) for e in emb]
            l.insert(0, str(i))
            f.write("\t".join(l) + "\n")

    print("embedding saved!")