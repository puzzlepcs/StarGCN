'''
Embedding nodes of toy examples using node2vec
'''


import argparse
import inspect, os.path

import scipy.sparse as sp
import networkx as nx
from gensim.models import Word2Vec

from src.node2vec import Graph
from data.data_helper import load_txt
from src.utils import generate_incidence_matrix, generate_clique_adjacency_matrix, generate_adjacency_matrix
from src.utils import ensure_dir


def parse_args():
    '''
    Parses the node2vec arguments
    :return:
    :rtype:
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument(
        '--data', nargs='?', default='121', help='Input graph.'
    )
    parser.add_argument(
        '--dimensions', type=int, default=16, help='Number of dimensions. Default is 16.'
    )
    parser.add_argument(
        '--walk-length', type=int, default=8, help='Length of walk per source. Default is 80.'
    )
    parser.add_argument(
        '--num-walks', type=int, default=2, help='Number of walks per source. Default is 10.'
    )
    parser.add_argument(
        '--window-size', type=int, default=2, help='Context size of optimization. Default is 10.'
    )
    parser.add_argument(
        '--iter', default=1, type=int, help='Number of epochs in SGD.'
    )
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers. Default is 1.')
    parser.add_argument(
        '--p', type=float, default=1, help='Return hyyperparameter. Default is 1.'
    )
    parser.add_argument(
        '--q', type=float, default=1, help='Inout hyperparameter. Default is 1.'
    )

    parser.add_argument('--clique', dest='clique', action='store_true',
                        help='Boolean specifying clique(star)expansion. Default is Star expansion')
    parser.set_defaults(clique=False)

    return parser.parse_args()


def read_graph():
    '''
    Read the input network in networkx
    '''
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    d = os.path.join(current, 'data', 'toy')

    hypergraph, n_node, m_hyperedge = load_txt(d, f'{args.data}.txt')
    inc_mat = generate_incidence_matrix(hypergraph, n_node, m_hyperedge)

    if args.clique:
        adj_mat = generate_clique_adjacency_matrix(hypergraph, n_node)
    else:
        adj_mat = generate_adjacency_matrix(inc_mat, n_node, m_hyperedge)

    G = nx.from_scipy_sparse_matrix(adj_mat, create_using=nx.Graph)
    return G


def learn_embeddings(walks):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=args.dimensions, window=args.window_size,
        min_count=0, sg=1, workers=args.workers, epochs=args.iter)

    return model.wv


def main(args):
    nx_G = read_graph()
    G = Graph(nx_G, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    word_vectors = learn_embeddings(walks)

    # Save created node embeddings
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    expansion = "clique" if args.clique else "star"
    print(expansion)
    file_name = f"{args.data}_{expansion}_d{args.dimensions}.emb"

    emb_file = os.path.join(current, "emb", "toy", file_name)
    ensure_dir(emb_file)

    word_vectors.save_word2vec_format(emb_file)


if __name__=="__main__":
    args = parse_args()
    main(args)
