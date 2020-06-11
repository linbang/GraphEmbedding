import pickle

import numpy as np


a = '../data/author_label.txt'
p = '../data/paper.txt'
t = '../data/term.txt'
c = '../data/conf.txt'
pa = '../data/paper_author.txt'
pc = '../data/paper_conf.txt'


a_list = set()
labels = []
with open(a, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        a_list.add(str('A' + line[0]))
        labels.append(int(line[1]))
p_list = set()
with open(pa, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('A' + line[1])
        if n2 in a_list:
            p_list.add(n1)
t_list = set()
with open(t, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        t_list.add(str('T' + line[0]))
c_list = set()
with open(pc, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('C' + line[1])
        if n1 in p_list:
            c_list.add(n2)
num_author = len(a_list)
num_paper = len(p_list)
num_conf = len(c_list)
num_term = len(t_list)

print(num_author)
print(num_paper)
print(num_conf)
print(num_term)


# 边关系提取

pa_list = []
with open(pa, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('A' + line[1])
        if n1 in p_list and n2 in a_list:
            trip = tuple([list(p_list).index(n1), list(a_list).index(n2)])
            pa_list.append(trip)
pc_list = []
with open(pc, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        n1 = str('P' + line[0])
        n2 = str('C' + line[1])
        if n1 in p_list and n2 in c_list:
            trip = tuple([list(p_list).index(n1), list(c_list).index(n2)])
            pc_list.append(trip)

print(len(pa_list))
print(len(pc_list))


# features
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
'''
paper_features = np.zeros(shape=(num_paper, num_term))
count = 0
for k,v in pt_dict.items():
    for term in v:
        paper_features[count][t_list.index(term)] = 1
    count += 1
'''

# author的feature是其写过的paper的所有term
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

'''
# conf的feature是与之有关的所有paper的关键词
pc = './paper_conf.txt'
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
count = 0
for k,v in cp_dict.items():
    for paper in v:
        for term in pt_dict[paper]:
            conf_features[count][t_list.index(term)] = 1
    count += 1
'''
node_list = list(a_list) + list(p_list) + list(c_list)

with open('../output/DBLP_HAN.pickle', 'wb') as f:
    pickle.dump((a_list, p_list, c_list, node_list), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((pa_list, pc_list), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(author_features, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

print('finish dump data')
