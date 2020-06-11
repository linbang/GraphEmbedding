import torch
from sklearn.metrics import f1_score

import argparse

from han_model import HAN
from han_utils import load_dblp, EarlyStopping, load_dblp_labels, load_amazon
from Linear_evaluation import evaluate_embeddings
from Amazon_evaluation import evaluate_amazon
from gcn_train import nce_loss

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


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        _, logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def get_embeddings(embeddings, node_list):
    _embeddings = {}
    embeddings = embeddings.tolist()
    for index, emb in enumerate(embeddings):
        _embeddings[node_list[index]] = emb
    return _embeddings


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_mask, val_mask, test_mask, node_list = load_dblp()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    print(features.shape)
    print("finish loading data")

    if args['hetero']:
        model = HAN(meta_paths=[['ap', 'pa'], ['ap', 'pc', 'cp', 'pa']],
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
    else:
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        _, logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

    h, _ = model(g, features)
    print(h.shape)

    embeddings = get_embeddings(h, node_list)
    labels = load_dblp_labels()
    X = list(labels.keys())
    Y = list(labels.values())

    for p in [0.2, 0.4, 0.6, 0.8]:
        evaluate_embeddings(p, embeddings, X, Y)


def train_amazon():
    num_walks = 5
    walk_length = 5
    window_size = 5
    neighbor_samples = 5
    hidden_uints = 32
    num_heads = [8]
    dropout = 0.5
    out_size = 200
    lr = 0.1
    weight_decay = 0.001
    epochs = 5

    g, features, train_pairs, neg_neighbors, vocab, edge_data_by_type = load_amazon(num_walks, neighbor_samples,
                                                                                    window_size)
    features = torch.FloatTensor(features)
    print("特征的shape是：", features.shape)

    metapaths = []
    for edge_type in edge_data_by_type.keys():
        metapaths.append([edge_type] * walk_length)
    model = HAN(meta_paths=metapaths,
                in_size=features.shape[1],
                hidden_size=hidden_uints,
                out_size=out_size,
                num_heads=num_heads,
                dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        _, h = model(g, features)
        loss = nce_loss(h, train_pairs, neg_neighbors)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, h = model(g, features)
    embeddings = get_embeddings(h, list(vocab))
    evaluate_amazon(embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('--hetero', default=True)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-heads', default=[10])
    parser.add_argument('--hidden_units', default=20)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--weight_decay', default=0.001)
    parser.add_argument('--num_epochs', default=10)
    parser.add_argument('--patience', default=20)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--device', default='cuda: 0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args().__dict__
    # main(args)
    train_amazon()
