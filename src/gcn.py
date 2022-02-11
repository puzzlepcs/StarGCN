import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.utils import sparse_mx_to_torch_sparse_tensor, normalize

class GraphConvolutionLayer(torch.nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, latent_dim:int, weight:bool=True, non_linear:bool=True) -> None:
        super(GraphConvolutionLayer, self).__init__()

        self.latent_dim = latent_dim
        self.use_weight = weight
        self.non_linear = non_linear

        if weight:
            self.weight = Parameter(torch.FloatTensor(latent_dim, latent_dim))
            self.bias = Parameter(torch.FloatTensor(latent_dim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.use_weight:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        if self.use_weight:
            x = torch.spmm(x, self.weight) + self.bias
        out = torch.spmm(G, x)

        if not self.non_linear:
            out = F.relu(out)

        return out

    def __repr__(self):
        return self.__class__.__name__ + " (" \
            + "weight " + str(self.use_weight) + ", non-linear " + str(self.non_linear)



class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def star_(self, node_emb, hyperedge_emb, h, X):
        """
        Compute star scores of given hyperedge candidates
        :param node_emb: embeddings of nodes
        :type node_emb:
        :param hyperedge_emb: embeddings of hyperedges
        :type hyperedge_emb: torch Tensor
        :param h: hyperege id list
        :type h: torch Tensor
        :param X: hyperedge list (containing node sets)
        :type X: torch Tensor
        :return: Tensor containing star scores of the given hyperedge candidates
        :rtype: torch Tensor
        """
        out = torch.zeros_like(h, dtype=torch.float32)
        for i, src in enumerate(h):
            src_emb = hyperedge_emb[src].expand(1, -1)
            nodes = X[i]
            embs = node_emb[nodes]

            scores = torch.sigmoid(torch.mm(src_emb, embs.T))
            scores = torch.min(scores, dim=1).values

            out[i] = scores
        return out.unsqueeze(dim=-1)

    def clique_(self, node_emb, X):
        """
        Compute clique scores of given hyperedge candidates
        :param node_emb: embeddings of nodes
        :type node_emb:
        :param X: hyperedge list (containing node sets)
        :type X: torch Tensor
        :return: Tensor containing clique scores of the given hyperedge candidates
        :rtype: torch Tensor
        """
        l = len(X)
        out = torch.zeros(l, dtype=torch.float32, device=self.cfg["device"])
        for i, nodes in enumerate(X):
            embs = node_emb[nodes]

            scores = torch.sigmoid(torch.mm(embs, embs.T))
            score = torch.min(scores)

            out[i] = score
        return out.unsqueeze(dim=-1)


class StarGCN(BasicModel):
    def __init__(self, cfg:dict, data:dict):
        super(StarGCN, self).__init__()
        self.cfg = cfg
        self.data = data

        d = self.data["input_dim"]
        c = self.data["num_class"]
        h = self.cfg["latent_dim"]
        l = self.cfg["num_layers"]
        device = self.cfg["device"]
        weight = self.cfg["weight"]
        non_linear = self.cfg["non_linear"]

        # Layer for dimensionality reduction (d->h)
        self.reduction_layer = torch.nn.Linear(
            in_features=d, out_features=h, bias=True).to(device)

        # Convolutional layers
        self.convolutional_layers = torch.nn.ModuleList([
            GraphConvolutionLayer(h, weight=weight, non_linear=non_linear).to(device) for _ in range(l)
        ])

        # Layer of classification (h->c)
        self.classification_layer = torch.nn.Linear(
            in_features=h, out_features=c, bias=True).to(device)

        # Initial features
        self.features = torch.FloatTensor(self.data["features"]).to(device)

        # Adjacency matrix of star expansion
        norm_adj = normalize(self.data["adj_mat"]).tocsr()
        self.G = sparse_mx_to_torch_sparse_tensor(norm_adj).to(device)

    def computer(self):
        """
        Propagation method
        :return: embeddings of node and hyperedges
        :rtype: torch Tensors
        """
        l = self.cfg["num_layers"]
        do = self.cfg["dropout"]
        n, m = self.data["n_node"], self.data["m_hyperedge"]

        all_emb = self.reduction_layer(self.features)
        all_emb = F.dropout(all_emb, do, training=self.training)

        embs = [all_emb]
        for i, layer in enumerate(self.convolutional_layers):
            all_emb = layer.forward(all_emb, self.G)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)

        nodes, hyperedges = torch.split(out, [n, m])

        return nodes, hyperedges

    def forward(self):
        do = self.cfg["dropout"]
        nodes, _ = self.computer()

        Z = self.classification_layer(nodes)
        Z = F.dropout(Z, do, training=self.training)

        return F.log_softmax(Z, dim=1)