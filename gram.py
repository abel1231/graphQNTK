import util
import time
import numpy as np
import scipy
from os.path import join
import argparse
import os
from multiprocessing import Pool
from gntk import GNTK

parser = argparse.ArgumentParser(description='GNTK computation')
# several folders, each folder one kernel
parser.add_argument('--dataset', type=str, default="BZR",
                        help='name of dataset (default: COLLAB)')
parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of mlp layers')
parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
parser.add_argument('--scale', type=str, default='degree',
						help='scaling methods')
parser.add_argument('--jk', type=int, default=1,
						help='whether to add jk')
parser.add_argument('--out_dir', type=str, default="out",
                    help='output directory')
args = parser.parse_args()

continuous = False

if args.dataset in ['IMDBBINARY', 'COLLAB', 'IMDBMULTI', 'COLLAB']:
    # social network
    degree_as_tag = True
elif args.dataset in ['MUTAG', 'PROTEINS', 'PTC', 'NCI1']:
    # bioinformatics
    degree_as_tag = False
elif args.dataset in ['BZR', 'COX2', 'ENZYMES', 'PROTEINS_full']:
    degree_as_tag = False
    continuous = True
elif args.dataset in ['Mutagenicity']:
    # model visualization
    degree_as_tag = False

util.print_config(args.dataset, args.num_mlp_layers, args.num_layers, args.jk)

graphs, _  = util.load_data(args.dataset, degree_as_tag)
labels = np.array([g.label for g in graphs]).astype(int)  # graph labels

if continuous:
    from sklearn.preprocessing import StandardScaler

    print("Dataset with continuous node attributes")
    node_features = np.concatenate([g.node_features for g in graphs], axis=0)
    sc = StandardScaler()
    sc.fit(node_features)
    for g in graphs:
        node_features = sc.transform(g.node_features)
        g.node_features = node_features / np.linalg.norm(node_features, axis=-1, keepdims=True).clip(min=1e-06)

gntk = GNTK(num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers, jk=args.jk, scale=args.scale)
A_list = []
diag_list = []  # store the sqrt(diag(sigma^{l}_{r})) from l = 1 to L-1 and r = 0 to R-1.   len = num of graphs ||| len x[.] = (L-1)*R ||| shape x[.][.] = num of nodes of the graph
diag_nngp_list = []
nngp_xx_list = []

# procesing the data
for i in range(len(graphs)): # traverse all graphs
    n = len(graphs[i].neighbors)  # num of nodes of the graph
    for j in range(n):
        graphs[i].neighbors[j].append(j)  # add the node itself to the list of neighbors
    edges = graphs[i].g.edges
    m = len(edges)

    row = [e[0] for e in edges]
    col = [e[1] for e in edges]

    A_list.append(scipy.sparse.coo_matrix(([1] * len(edges), (row, col)), shape = (n, n), dtype = np.float32))
    A_list[-1] = A_list[-1] + A_list[-1].T + scipy.sparse.identity(n) # add self-loop and make it symmetric
    diag, diag_nngp, nngp_xx= gntk.diag(graphs[i], A_list[i])
    diag_list.append(diag)
    diag_nngp_list.append(diag_nngp)
    nngp_xx_list.append(nngp_xx)



def calc(T):
    return gntk.gntk(graphs[T[0]], graphs[T[1]],
                     diag_list[T[0]], diag_list[T[1]],
                     A_list[T[0]], A_list[T[1]],
                     diag_nngp_list[T[0]], diag_nngp_list[T[1]],
                     nngp_xx_list[T[0]], nngp_xx_list[T[1]])

calc_list = [(i, j) for i in range(len(graphs)) for j in range(i, len(graphs))]

pool = Pool(80)
results = pool.map(calc, calc_list)

gram = np.zeros((len(graphs), len(graphs)))  # symmetric ntk matrix, shape: num of graphs by num of graphs
for t, v in zip(calc_list, results):
    gram[t[0], t[1]] = v
    gram[t[1], t[0]] = v
    

np.save(join(args.out_dir, 'gram'), gram)
np.save(join(args.out_dir, 'labels'), labels)
