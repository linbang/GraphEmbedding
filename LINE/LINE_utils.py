import numpy as np
import networkx as nx
import pickle


def load_dblp():
    with open('../dataset/DBLP/output/DBLP_LINE.pickle', 'rb') as f:
        node_list = pickle.load(f)
        edge_list = pickle.load(f)

    print(len(node_list))
    print(len(edge_list))

    g = nx.DiGraph()
    g.add_edges_from(edge_list)

    return g, node_list

def load_Amazon():
    trainfile = '../dataset/Amazon/data/train.txt'
    node_pairs = []
    node_list = set()
    for line in open(trainfile).readlines():
        line = line.strip().split(' ')
        node1 = 'I' + line[1]
        node2 = 'I' + line[2]
        node_pairs.append([node1, node2])
        node_list.add(node1)
        node_list.add(node2)

    g = nx.Graph()
    g.add_edges_from(node_pairs)
    return g, list(node_list)

def load_dataset(dataset):
    if dataset=='DBLP':
        return load_dblp()
    if dataset=='Amazon':
        return load_Amazon()


def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
            (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx