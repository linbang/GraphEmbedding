import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from sklearn.model_selection import train_test_split
from gatne_utils import generate_vocab, get_graph

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def transpose(l):
    new_l = []
    for k in l:
        new_l.append(tuple([k[1], k[0]]))
    return new_l


def load_dblp(remove_self_loop):
    with open('../dataset/DBLP/output/DBLP_HAN.pickle', 'rb') as f:
        a_list, p_list, c_list, node_list = pickle.load(f)
        pa_list, pc_list = pickle.load(f)
        author_features = pickle.load(f)
        labels = pickle.load(f)

    # 构造异构网络
    pa = dgl.bipartite(pa_list, 'paper', 'pa', 'author')
    ap = dgl.bipartite(transpose(pa_list), 'author', 'ap', 'paper')
    pc = dgl.bipartite(pc_list, 'paper', 'pc', 'conf')
    cp = dgl.bipartite(transpose(pc_list), 'conf', 'cp', 'paper')
    hg = dgl.hetero_from_relations([pa, ap, pc, cp])

    features = torch.FloatTensor(author_features)
    labels = torch.LongTensor(labels)

    num_class = 4

    alls = [i for i in range(len(a_list))]
    train_idx, x, _, _ = train_test_split(alls, labels, test_size=0.2, random_state=52)
    eval_idx, test_idx, _, _ = train_test_split(x, _, test_size=0.5, random_state=40)

    num_nodes = hg.number_of_nodes('author')
    train_mask = get_binary_mask(num_nodes, train_idx)
    eval_mask = get_binary_mask(num_nodes, eval_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return hg, features, labels, num_class, train_mask, test_mask, eval_mask, node_list


def generate_pairs(all_walks, window_size):
    # for each node, choose the first neighbor and second neighbor of it to form pairs
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((walk[i], walk[i - j]))
                    if i + j < len(walk):
                        pairs.append((walk[i], walk[i + j]))
    return pairs


def load_amazon(num_walks, neighbor_samples, window_size):
    trainfile = '../dataset/Amazon/data/train.txt'
    node_list = set()
    edge_data_by_type = dict()
    for line in open(trainfile, 'r').readlines():
        line = line.strip().split(' ')
        if line[0] not in edge_data_by_type:
            edge_data_by_type[line[0]] = list()
        x, y = str('I' + line[1]), str('I' + line[2])
        edge_data_by_type[line[0]].append((x, y))
        edge_data_by_type[line[0]].append((x, y))
        node_list.add(x)
        node_list.add(y)

    node_list = list(node_list)
    print("all node number is:", len(node_list))
    print("all edge type is:", edge_data_by_type.keys())

    index2word, vocab, type_nodes = generate_vocab(edge_data_by_type)
    edge_types = list(edge_data_by_type.keys())
    num_nodes = len(index2word)
    edge_type_count = len(edge_types)

    g = get_graph(edge_data_by_type, vocab)
    all_walks = []
    for i in range(edge_type_count):
        nodes = torch.LongTensor(type_nodes[i] * num_walks)  # 可以理解为每个node采样20个path
        traces, types = dgl.sampling.random_walk(g, nodes, metapath=[edge_types[i]] * (neighbor_samples - 1))  # 按照边的类型进行采样
        traces = traces.tolist()
        all_walks.append(traces)

    # 将训练序列拆成训练数据对, 得到邻居节点。
    train_pairs = generate_pairs(all_walks, window_size)  # window_size内的都是正样本
    neighbors = [[] for _ in range(num_nodes)]
    neg_neighbors = [[] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        a = edge_data_by_type[edge_types[r]]
        for (x, y) in a:
            ix = vocab[x]
            iy = vocab[y]
            neighbors[ix].append(iy)
            neighbors[iy].append(ix)
    for index in range(num_nodes):
        while len(neg_neighbors[index]) < 5:
            neg = np.random.choice(list(vocab.values()), 1)[0]
            if neg not in neg_neighbors[index] and neg not in neighbors[index]:
                neg_neighbors[index].append(neg)

    # 特征提取
    features = [[] for _ in range(num_nodes)]
    featurefile = '../dataset/Amazon/data/feature.txt'
    feature_dim = 0
    for line in open(featurefile).readlines():
        line = line.strip().split(' ')
        node = 'I' + line[0]
        fea = [float(x) for x in line[1:]]
        if node in vocab.keys():
            features[vocab[node]] = fea
            feature_dim = len(fea)
    for k, v in enumerate(features):
        if len(v) == 0:
            features[k] = [0.0] * feature_dim

    return g, features, train_pairs, neg_neighbors, vocab, edge_data_by_type


def load_dblp_labels():
    f1 = '../dataset/DBLP/data/author_label.txt'

    labels = {}
    for line in open(f1).readlines():
        line = line.strip().split('\t')
        node = str('A' + line[0])
        labels[node] = line[1]

    return labels


def load_data(dataset, remove_self_loop=False):
    if dataset == 'DBLP':
        return load_dblp(remove_self_loop)


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
