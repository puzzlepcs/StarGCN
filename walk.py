import argparse
import inspect
import os.path

import scipy.sparse as sp
import networkx as nx
from gensim.models import Word2Vec

from src.node2vec import Graph
from data.data_helper import load_pickle
from src.utils import generate_incidence_matrix, generate_adjacency_matrix, ensure_dir


def parse_args():
    '''
    Parses the node2vec arguments
    :return:
    :rtype:
    '''

    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument(
        '--data', nargs='?', default='citeseer', help='Input graph.')
    parser.add_argument(
        '--dimensions', type=int, default=32, help='Number of dimensions. Default is 32.')
    parser.add_argument(
        '--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    parser.add_argument(
        '--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')
    parser.add_argument(
        '--window-size', type=int, default=10, help='Context size of optimization. Default is 10.')
    parser.add_argument(
        '--iter', default=1, type=int, help='Number of epochs in SGD.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers. Default is 1.')
    parser.add_argument(
        '--p', type=float, default=1, help='Return hyyperparameter. Default is 1.')
    parser.add_argument(
        '--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--clique', dest='clique', action='store_true',
                        help='Boolean specifying clique(star)expansion. Default is Star expansion')
    parser.set_defaults(clique=False)

    return parser.parse_args()


def read_graph():
    '''
    Read the input network in networkx.
    :return:
    :rtype:
    '''
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    d = os.path.join(current, 'data', args.data)

    hypergraph = load_pickle(d, 'hypergraph.pickle')
    labels = load_pickle(d, 'labels.pickle')

    m_hyperedge = len(hypergraph)
    n_node = len(labels)

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


    # G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
    G = nx.from_scipy_sparse_matrix(adj_mat, create_using=nx.Graph)

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    :param walks:
    :type walks:
    :return:
    :rtype:
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=args.dimensions, window=args.window_size,
        min_count=0, sg=1, workers=args.workers, epochs=args.iter)

    return model.wv


def main(args):
    '''

    :param args:
    :type args:
    :return:
    :rtype:
    '''
    nx_G = read_graph()
    G = Graph(nx_G, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    word_vectors = learn_embeddings(walks)

    # save node embeddings
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    expansion = "clique" if args.clique else "star"
    emb_file = os.path.join(
        current, "emb", args.data, f"walk_{expansion}_d{args.dimensions}.emb")
    ensure_dir(emb_file)

    word_vectors.save_word2vec_format(emb_file)

if __name__=="__main__":
    args = parse_args()
    main(args)