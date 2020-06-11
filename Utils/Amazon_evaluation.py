import numpy as np
from sklearn import metrics

def cosine_similarity(x, y):
    t1 = x.dot(y.T)
    t2 = np.linalg.norm(x) * np.linalg.norm(y)
    return t1 / t2

def cal_metrics(preds, labels):
    orc_auc = metrics.roc_auc_score(labels, preds)
    y_pred = [1 if i > 0.5 else 0 for i in preds]
    f1 = metrics.f1_score(labels, y_pred)
    ap = metrics.average_precision_score(labels, y_pred)

    ps, rs, _ = metrics.precision_recall_curve(labels, preds)
    pr_auc = metrics.auc(rs, ps)
    print('ROC-AUC:', orc_auc)
    print('F1:', f1)
    print('AP:', ap)
    print('PR-AUC', pr_auc)

def evaluate_amazon(embeddings):
    testfile = '../dataset/Amazon/data/test.txt'
    labels = []
    preds = []
    for line in open(testfile).readlines():
        line = line.strip().split(' ')
        node1 = str('I' + line[1])
        node2 = str('I' + line[2])
        label = int(line[3])
        if node1 in embeddings.keys() and node2 in embeddings.keys():
            emb1 = np.array(embeddings[node1])
            emb2 = np.array(embeddings[node2])
            score = cosine_similarity(emb1, emb2)
            preds.append(score)
            labels.append(label)

    cal_metrics(preds, labels)