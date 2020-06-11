from __future__ import print_function


import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression


def evaluate_DBLP(embeddings):
    labels = load_dblp_labels()
    X = list(labels.keys())
    Y = list(labels.values())
    for tr_frac in [0.2, 0.4, 0.6, 0.8]:
        evaluate_embeddings(tr_frac, embeddings, X, Y)


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y,Y_)
        #print('-------------------')
        print(results)
        print('-------------------')
        return results


    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def evaluate_embeddings(tr_frac, embeddings, X, Y):
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)

def load_dblp_labels():
    f1 = '../dataset/DBLP/data/author_label.txt'
    f2 = '../dataset/DBLP/data/paper_label.txt'
    f3 = '../dataset/DBLP/data/conf_label.txt'

    labels = {}
    for line in open(f1).readlines():
        line = line.strip().split('\t')
        node = str('A' + line[0])
        labels[node] = line[1]

    for line in open(f2).readlines():
        line = line.strip().split('\t')
        node = str('P' + line[0])
        labels[node] = line[1]

    for line in open(f3).readlines():
        line = line.strip().split('\t')
        node = str('C' + line[0])
        labels[node] = line[1]

    return labels