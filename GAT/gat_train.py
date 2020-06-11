import torch
from sklearn.metrics import f1_score

from gat_utils import load_dblp, EarlyStopping
from gat_model import GAT, UN_GAT
from gcn_train import nce_loss
from gcn_utils import load_amazon

from Linear_evaluation import evaluate_DBLP
from Amazon_evaluation import evaluate_amazon


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
        h, logits = model(features)
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

    device = torch.device("cpu")

    epochs = 10
    hidden_dim = 16
    out_dim = 200
    num_heads = 8
    patience = 10
    lr = 0.1
    weight_decay = 0.01

    # create the dataset
    g, features, labels, num_class, train_mask, eval_mask, test_mask, node_list = load_dblp()

    # create model
    model = GAT(g, in_dim=features.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, num_heads=num_heads)

    stopper = EarlyStopping(patience=patience)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fcn = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        h, logits = model(features)
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
    evaluate_DBLP(embeddings)


def train_amazon():
    num_walks = 5
    walk_length = 5
    window_size = 5
    neighbor_samples = 5
    weight_decay = 0.001
    lr = 0.01
    epochs = 1
    hidden_dim = 16
    out_dim = 200
    num_heads = 8

    g, features, train_pairs, neg_neighbors, word2index, index2word = load_amazon(num_walks, walk_length, window_size,
                                                                                  neighbor_samples)
    features = torch.FloatTensor(features)

    model = UN_GAT(g, in_dim=features.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, num_heads=num_heads)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    device = torch.device("cpu")
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

    train_amazon()
