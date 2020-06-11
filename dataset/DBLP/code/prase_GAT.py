import pickle

import numpy as np


a = '../data/author_label.txt'
p = '../data/paper_label.txt'
c = '../data/conf_label.txt'

a_list = []
labels = []
with open(a, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        a_list.append(str('A' + line[0]))
        labels.append(int(line[1]))
p_list = []
with open(p, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        p_list.append(str('P' + line[0]))
        labels.append(int(line[1]))
c_list = []
with open(c, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        c_list.append(str('C' + line[0]))
        labels.append(int(line[1]))

node_list = a_list + p_list + c_list

num_author = len(a_list)
num_paper = len(p_list)
num_conf = len(c_list)

print(num_author)
print(num_paper)
print(num_conf)

# 边关系提取
pa = '../data/paper_author.txt'
pc = '../data/paper_conf.txt'
edge_list = []
with open(pa, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('A' + line[1])
        if n1 in p_list and n2 in a_list:
            trip = tuple([node_list.index(n1), node_list.index(n2)])
            edge_list.append(trip)
with open(pc, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('C' + line[1])
        if n1 in p_list and n2 in c_list:
            trip = tuple([node_list.index(n1), node_list.index(n2)])
            edge_list.append(trip)

print(len(edge_list))

# features
t = '../data/term.txt'
t_list = set()
with open(t, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        t_list.add(str('T' + line[0]))
num_term = len(t_list)

# paper的features就是terms
pt = '../data/paper_term.txt'
pt_dict = dict()
with open(pt, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = 'P' + str(line[0])
        n2 = 'T' + str(line[1])
        if n1 not in pt_dict:
            pt_dict[n1] = set()
        pt_dict[n1].add(n2)
paper_features = np.zeros(shape=(num_paper, num_term))
for index,paper in enumerate(p_list):
    for term in pt_dict[paper]:
        paper_features[index][list(t_list).index(term)] = 1

# author的feature是其写过的paper的所有term
pa = '../data/paper_author.txt'
ap_dict = dict()
with open(pa, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = 'P' + str(line[0])
        n2 = 'A' + str(line[1])
        if n2 not in ap_dict:
            ap_dict[n2] = set()
        ap_dict[n2].add(n1)
author_features = np.zeros(shape=(num_author, num_term))
for index,author in enumerate(a_list):
    for paper in ap_dict[author]:
        for term in pt_dict[paper]:
            author_features[index][list(t_list).index(term)] = 1

# conf的feature是与之有关的所有paper的关键词
pc = '../data/paper_conf.txt'
cp_dict = dict()
with open(pc, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = 'P' + str(line[0])
        n2 = 'C' + str(line[1])
        if n2 not in cp_dict:
            cp_dict[n2] = set()
        cp_dict[n2].add(n1)
conf_features = np.zeros(shape=(num_conf, num_term))
for index,conf in enumerate(c_list):
    for paper in cp_dict[conf]:
        for term in pt_dict[paper]:
            conf_features[index][list(t_list).index(term)] = 1

print(author_features.shape)
print(paper_features.shape)
print(conf_features.shape)

features = np.concatenate((author_features, paper_features, conf_features), axis=0)
print(len(node_list))
print(features.shape)


with open('../output/DBLP_GAT.pickle', 'wb') as f:
    pickle.dump(node_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(edge_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

print('finish dump data')
