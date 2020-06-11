import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
import dgl
import datetime

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_dblp():
    with open('../dataset/DBLP/output/DBLP_GAT.pickle', 'rb') as f:
        node_list = pickle.load(f)
        edge_list = pickle.load(f)
        features = pickle.load(f)
        labels = pickle.load(f)

    num_nodes = len(node_list)

    alls = [i for i in range(num_nodes)]
    train_idx, x, _, _ = train_test_split(alls, labels, test_size=0.2, random_state=52)
    eval_idx, test_idx, _, _ = train_test_split(x, _, test_size=0.5, random_state=40)

    train_mask = get_binary_mask(num_nodes, train_idx)
    eval_mask = get_binary_mask(num_nodes, eval_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    num_class = 4

    #构造图
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, des = tuple(zip(*edge_list))
    g.add_edges(src, des)
    g.add_edges(des, src)

    print(num_nodes)
    print(features.shape)

    return g, features, labels, num_class, train_mask, eval_mask, test_mask, node_list

def load_training_data(f_name):
    print('We are loading data from:', f_name)
    edge_data = []
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')  # line[-1] == '\n'
            x, y = 'I' +words[1], 'I' +words[2]
            edge_data.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data

def generate_vocab(network_data):
    node1, node2 = zip(*network_data)
    node_list = list(node1) + list(node2)
    node_list = list(set(node_list))

    word2index = {node_list[i]:i for i in range(len(node_list))}
    index2word = {i:node_list[i] for i in range(len(node_list))}

    return node_list, index2word, word2index

def get_graph(network_data, word2index):
    num_nodes = len(word2index)
    edges = []
    for edge in network_data:
        edges.append((word2index[edge[0]], word2index[edge[1]]))
        edges.append((word2index[edge[1]], word2index[edge[0]]))
    g = dgl.graph(edges, num_nodes=num_nodes)
    return g


def generate_pairs(all_walks, window_size):
    # for each node, choose the first neighbor and second neighbor of it to form pairs
    pairs = []
    skip_window = window_size // 2
    for walk in all_walks:
        for i in range(len(walk)):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    pairs.append((walk[i], walk[i - j]))
                if i + j < len(walk):
                    pairs.append((walk[i], walk[i + j]))

    return pairs

def load_amazon(num_walks, walk_length, window_size, neighbor_samples):
    trainfile = '../dataset/Amazon/data/train.txt'
    network_data = load_training_data(trainfile)
    node_list, index2word, word2index = generate_vocab(network_data)
    # 随机游走，得到训练序列
    g = get_graph(network_data, word2index)
    # 随机游走
    traces, types = dgl.sampling.random_walk(g, list(index2word.keys()) * num_walks, length=walk_length - 1)  # 按照边的类型进行采样
    walks = traces.tolist()
    #正样本
    train_pairs = generate_pairs(walks, window_size)  # window_size内的都是正样本
    #负样本
    neg_neighbors =[[] for _ in range(len(word2index))]
    neighbors = [[] for _ in range(len(word2index))]
    for (x, y) in train_pairs:
        neighbors[x].append(y)
        neighbors[y].append(x)
    for index in range(len(word2index)):
        while len(neg_neighbors[index]) < 5:
            neg = np.random.choice(list(word2index.values()), 1)[0]
            if neg not in neg_neighbors[index] and neg not in neighbors[index]:
                neg_neighbors[index].append(neg)

    features = [[] for _ in range(len(word2index))]
    featurefile = '../dataset/Amazon/data/feature.txt'
    feature_dim = 0
    for line in open(featurefile).readlines():
        line = line.strip().split(' ')
        node = 'I'+line[0]
        fea = [float(x) for x in line[1:]]
        if node in word2index.keys():
            features[word2index[node]] = fea
            feature_dim = len(fea)
    for k,v in enumerate(features):
        if len(v) == 0:
            features[k] = [0.0] * feature_dim

    return g, features, train_pairs, neg_neighbors, word2index, index2word


#load_amazon(5,5,5,5)

def load_data(args):
    if args.dataset == 'DBLP':
        return load_dblp()
    if args.dataset == 'Amazon':
        return load_amazon()


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