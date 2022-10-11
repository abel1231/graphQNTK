import networkx as nx
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from data_io import load_graphdata

DATA = ['Mutagenicity', 'BZR', 'COX2', 'ENZYMES', 'FRANKENSTEIN', 'PROTEINS_full']

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy float tensor, one-hot representation of the tag that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label # label index
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0

        self.max_neighbor = 0  # max degree


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')

    if dataset in DATA:
        return load_graphdata(dataset, datadir='./dataset')
    g_list = []
    label_dict = {}  # dict of graph labels: true label --> label index
    feat_dict = {}   # dict of node labels: true label --> label index

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())  # num of graphs
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]  # n: num of nodes of the graph  l: the graph label
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0 # num of edges of the graph
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])  # useless?
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped # dict of node labels: true label --> label index
                node_tags.append(feat_dict[row[0]])  # list of node labels of the graph, len(node_tags) = n

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

            
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

    # if the dataset has no node labels, use node degree as node label. Otherwise
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree(range(len(g.g)))).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros([len(g.node_tags), len(tagset)])
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1  # one-hot node feature matrix of the graph, num_nodes by num_node_classes


    print('# num of graph classes: %d' % len(label_dict))
    print('# num of node classes: %d' % len(tagset))

    print("# num of graph: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def print_config(dataset, num_mlp_layers, num_layers, jk):
    print('########## Configuration ##########')
    print('           dataset: %s' % dataset)
    print('           num_layers: %d' % num_layers)
    print('           num_mlp_layers: %d' % num_mlp_layers)
    print('           jk: %d' % jk)