import numpy as np
import torch
from sklearn.metrics import f1_score
from gcn_utils import load_dblp, EarlyStopping, load_amazon
from gcn_model import GCN, UN_GCN
import torch.nn.functional as F

from Linear_evaluation import evaluate_DBLP
from Amazon_evaluation import evaluate_amazon

import warnings

warnings.filterwarnings("ignore")


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        _, logits = model(features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def get_embeddings(embeddings, node_list):
    _embeddings = {}
    embeddings = embeddings.tolist()
    for index, emb in enumerate(embeddings):
        _embeddings[node_list[index]] = emb
    return _embeddings


def train_dblp():
    epochs = 10
    hidden_dim = 128
    n_layers = 2
    dropout = 0.5
    patience = 20
    lr = 0.01
    weight_decay = 0.001

    # create the dataset
    g, features, labels, num_class, train_mask, eval_mask, test_mask, node_list = load_dblp()

    # create model
    model = GCN(g,
                in_feats=features.shape[1],
                n_hidden=hidden_dim,
                n_classes=num_class,
                n_layers=n_layers,
                activation=F.relu,
                dropout=dropout)

    stopper = EarlyStopping(patience=patience)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fcn = torch.nn.CrossEntropyLoss()

    device = torch.device("cpu")

    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        _, logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, features, labels, eval_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

    h, _ = model(features)
    embeddings = get_embeddings(h, node_list)
    evaluate_DBLP(h)


def cal(emb1, emb2):
    x = torch.mm(emb1, emb2.t())
    return torch.sigmoid(x)


def nce_loss(embeddings, train_pairs, neg_neighbors):
    losses = []
    for idx1, idx2 in train_pairs:
        emb1 = torch.reshape(embeddings[idx1], (1, 200))
        pos_emb = torch.reshape(embeddings[idx2], (1, 200))
        pos_dot = cal(emb1, pos_emb)
        neg_dots = []
        for neg_idx in neg_neighbors[idx1]:
            neg_emb = torch.reshape(embeddings[neg_idx], (1, 200))
            neg_dots.append(cal(emb1, neg_emb))
        neg_dot = -sum(neg_dots) / len(neg_dots)
        losses.append(-(pos_dot + neg_dot))
    result = sum(losses) / len(losses)
    return result


def train_Amazon():
    num_walks = 5
    walk_length = 5
    window_size = 5
    neighbor_samples = 5
    n_hidden = 200
    n_layers = 3
    activation = F.relu
    dropout = 0.5
    weight_decay = 0.001
    lr = 0.01
    epochs = 1

    g, features, train_pairs, neg_neighbors, word2index, index2word = load_amazon(num_walks, walk_length, window_size,
                                                                                  neighbor_samples)
    neg_features = []
    for index, neighbors in enumerate(neg_neighbors):
        neg_features.append([features[neighbor] for neighbor in neighbors])

    features = torch.FloatTensor(features)
    neg_features = torch.FloatTensor(neg_features)
    print(features.shape)
    print(neg_features.shape)

    device = torch.device("cpu")
    model = UN_GCN(g=g,
                   in_feats=features.shape[1],
                   n_hidden=n_hidden,
                   n_layers=n_layers,
                   activation=activation,
                   dropout=dropout
                   )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        h = model(features)
        loss = nce_loss(h, train_pairs, neg_neighbors)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    h = model(features)
    embeddings = get_embeddings(h, list(word2index))
    evaluate_amazon(embeddings)


if __name__ == '__main__':
    # train_dblp()
    train_Amazon()
