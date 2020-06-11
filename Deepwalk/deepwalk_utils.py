from __future__ import print_function

import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import dgl
import networkx as nx


def load_dblp(walk_length, num_walks):


    with open('../dataset/DBLP/output/DBLP_Deepwalk.pickle', 'rb') as f:
        node_list = pickle.load(f)
        edge_list = pickle.load(f)
        labels = pickle.load(f)

    word2index = {node_list[i]:i for i in range(len(node_list))}
    index2word = {i:node_list[i] for i in range(len(node_list))}

    # 构造图
    src, des = tuple(zip(*edge_list))
    node1s = src + des
    node2s = des + src
    g = dgl.graph((node1s, node2s))

    # 随机游走
    traces, types = dgl.sampling.random_walk(g, list(index2word.keys()) * num_walks, length=walk_length - 1)  # 按照边的类型进行采样
    sentences = []
    for s in traces.tolist():
        t = []
        for i in s:
            t.append(index2word[i])
        sentences.append(t)

    return g, sentences, word2index, index2word


def load_amazon(num_walks, walk_length):
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

    node_list = list(node_list)
    word2index = {node_list[i]: i for i in range(len(node_list))}
    index2word = {i: node_list[i] for i in range(len(node_list))}
    edges = []
    for n1, n2 in node_pairs:
        edges.append([word2index[n1], word2index[n2]])
    # 构造图
    g = nx.Graph()
    g.add_edges_from(edges)
    G = dgl.graph(g)
    # 随机游走
    traces, types = dgl.sampling.random_walk(G, list(word2index.values()) * num_walks, length=walk_length - 1)  # 按照边的类型进行采样
    sentences = []
    for s in traces.tolist():
        t = []
        for i in s:
            t.append(index2word[i])
        sentences.append(t)

    return G, sentences, word2index, index2word


def load_dataset(dataset, num_walks, walk_length):
    if dataset == 'DBLP':
        return load_dblp(num_walks, walk_length)
    if dataset == 'Amazon':
        return load_amazon(num_walks, walk_length)

#load_amazon(5,5)
