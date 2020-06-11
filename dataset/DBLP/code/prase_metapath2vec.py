import pickle

import numpy as np


a = '../data/author.txt'
p = '../data/paper.txt'
c = '../data/conf.txt'

a_list = []
#labels = {}
with open(a, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        a_list.append(str('A' + line[0]))
        #labels[str('A' + line[0])] = line[1]
p_list = []
with open(p, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        p_list.append(str('P' + line[0]))
        #labels[str('P' + line[0])] = line[1]
c_list = []
with open(c, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        c_list.append(str('C' + line[0]))
        #labels[str('C' + line[0])] = line[1]

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
pa_list = []
nodes = set()
with open(pa, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('A' + line[1])
        if n1 in p_list and n2 in a_list:
            trip = tuple([list(node_list).index(n1), list(node_list).index(n2)])
            pa_list.append(trip)
pc_list = []
with open(pc, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('C' + line[1])
        if n1 in p_list and n2 in c_list:
            trip = tuple([list(node_list).index(n1), list(node_list).index(n2)])
            pc_list.append(trip)

with open('../output/DBLP_Metapath2vec.pickle', 'wb') as f:
    pickle.dump((a_list, p_list, c_list, node_list), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((pa_list, pc_list), f, pickle.HIGHEST_PROTOCOL)
    #pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

print('finish dump data')