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