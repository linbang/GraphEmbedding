import pickle

import numpy as np


a = '../data/author.txt'
p = '../data/paper.txt'
c = '../data/conf.txt'

a_list = []
labels = []
with open(a, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        a_list.append(str('A' + line[0]))
        labels.append(line[1])
p_list = []
with open(p, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        p_list.append(str('P' + line[0]))
        labels.append(line[1])
c_list = []
with open(c, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        c_list.append(str('C' + line[0]))
        labels.append(line[1])

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
nodes = set()
with open(pa, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('A' + line[1])
        if n1 in p_list and n2 in a_list:
            trip = tuple([n1, n2])
            edge_list.append(trip)
            nodes.add(n1)
            nodes.add(n2)
with open(pc, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('C' + line[1])
        if n1 in p_list and n2 in c_list:
            trip = tuple([n1, n2])
            edge_list.append(trip)
            nodes.add(n1)
            nodes.add(n2)

nodes = list(nodes)
edges = []
for n1,n2 in edge_list:
    edges.append(tuple([nodes.index(n1), nodes.index(n2)]))

label_f = []
for w in nodes:
    label_f.append(labels[node_list.index(w)])

print(len(edge_list))
print(len(edges))
print(len(nodes))
print(len(node_list))
print(len(label_f))

with open('../output/DBLP_Deepwalk.pickle', 'wb') as f:
    pickle.dump(nodes, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(edges, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(label_f, f, pickle.HIGHEST_PROTOCOL)

print('finish dump data')