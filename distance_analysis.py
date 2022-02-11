'''
RQ 1, RQ 2) effectiveness of Strategy 1 & Strategy 2
'''

import argparse
import inspect
import os.path

import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial import distance
import pandas as pd

from src.analyze import intra_distance, inter_distance
from src.analyze import get_hyperedge_pairs
from data.data_helper import load_pickle, load_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Get .")

    parser.add_argument(
        '--data', nargs='?', default='citeseer', help='Input graph.')
    parser.add_argument('--emb', default='walk_clique_d32.emb', help='Input embedding file.')

    return parser.parse_args()


def read_graph():
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    d = os.path.join(current, 'data', args.data)

    hypergraph = load_pickle(d, 'hypergraph.pickle')

    emb_dir = os.path.join(current, 'emb', args.data)
    wv = load_embeddings(emb_dir, args.emb)
    # wv = KeyedVectors.load_word2vec_format(emb_dir, binary=False)

    # keys = sorted([int(key) for key in wv.index_to_key])
    keys = sorted([int(key) for key in wv.keys()])
    embeddings = [wv[key] for key in keys]
    # embeddings = np.stack(embeddings)

    print(wv[0])
    print(embeddings[0])

    return hypergraph, embeddings


def print_quantiles(dists: list):
    median = np.median(dists)
    Q1 = np.quantile(dists, 0.25)
    Q3 = np.quantile(dists, 0.75)
    iqr = Q3 - Q1

    mmin = max(np.min(dists), Q1 - 1.5 * iqr)
    mmax = min(np.max(dists), Q3 + 1.5 * iqr)

    print("max", mmax)
    print("Q3", Q3)
    print("median", median)
    print("Q1", Q1)
    print("min", mmin)


def main():
    hypergraph, embeddings = read_graph()

    print(len(hypergraph), len(embeddings))

    dists = distance.cdist(embeddings, embeddings)
    hyperedge_pairs = get_hyperedge_pairs(hypergraph)

    intra_dists = intra_distance(hypergraph, hyperedge_pairs, dists)
    inter_dists = inter_distance(hypergraph, hyperedge_pairs, dists)

    # d = {'intra': intra_dists, 'inter': inter_dists}
    # df = pd.DataFrame(data=d)
    #
    # current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # emb_name = ''.join(args.emb.split('.')[:-1])
    # df_dir = os.path.join(current, 'result', args.data, f'{emb_name}.csv')
    # df.to_csv(df_dir)
    #
    # df = pd.read_csv(df_dir)
    # intra_dists = df['intra']
    # inter_dists = df['inter']

    intra_mean = np.mean(intra_dists)
    intra_std = np.std(intra_dists)
    intra_med = np.median(intra_dists)

    inter_mean = np.mean(inter_dists)
    inter_std = np.std(inter_dists)
    inter_med = np.median(inter_dists)

    print(f'[Intra dists] mean {intra_mean:.3f} med {intra_med:.3f} std {intra_std:.3f}')
    print(f'[Inter dists] mean {inter_mean:.3f} med {inter_med:.3f} std {inter_std:.3f}')

    #
    # print(" [intra_dists] ")
    # print_quantiles(intra_dists)
    # print(" [inter_dists] ")
    # print_quantiles(inter_dists)


if __name__=="__main__":
    args = parse_args()
    main()





