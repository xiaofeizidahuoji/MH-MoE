import argparse
import configparser
import torch
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

def loadGraph(adj_mx, hs, ls):
    graphwave = get_eigv(adj_mx+np.eye(adj_mx.shape[0]), hs)
    sampled_nodes_number = int(np.around(math.log(adj_mx.shape[0]))+2)*ls
    graph = csr_matrix(adj_mx)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    adj_gat = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]
    return adj_gat, graphwave


def parse_args(DATASET, args_base):
    # get configuration
    config_file = '../conf/STWave/{}.conf'.format(DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--device', type=str, default=config['general']['device'])

    # Data
    parser.add_argument('--seq_len', type=int, default=config['data']['seq_len'])
    parser.add_argument('--horizon', type=int, default=config['data']['horizon'])
    parser.add_argument('--output_dim', type=int, default=config['data']['output_dim'])

    # Model
    parser.add_argument('--input_dim', type=int, default=config['model']['input_dim'])
    parser.add_argument('--hidden_size', type=int, default=config['model']['hidden_size'])
    parser.add_argument('--layers', type=int, default=config['model']['layers'])
    parser.add_argument('--log_samples', type=int, default=config['model']['log_samples'])
    parser.add_argument('--time_in_day_size', type=int, default=config['model']['time_in_day_size'])
    parser.add_argument('--day_in_week_size', type=int, default=config['model']['day_in_week_size'])

    # Features Flags
    parser.add_argument('--wave_type', type=eval, default=config['model']['wave_type'])
    parser.add_argument('--wave_levels', type=int, default=config['model']['wave_levels'])


    args_predictor, _ = parser.parse_known_args()
    
    adj_mx = args_base.A_dict_np[DATASET]
    adjgat, gwv = loadGraph(adj_mx, 128, 1)
    
    args_predictor.adj_gat = adjgat
    args_predictor.gwv = gwv
    return args_predictor