import pickle
import dgl
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def transpose(l):
    new_l = []
    for k in l:
        new_l.append(tuple([k[1], k[0]]))
    return new_l

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

with open('../dataset/DBLP/DBLP.pickle', 'rb') as f:
    a_list, p_list, c_list = pickle.load(f)
    pa_list, pc_list = pickle.load(f)
    author_features = pickle.load(f)
    labels = pickle.load(f)

# 构造异构网络
pa = dgl.bipartite(pa_list, 'paper', 'pa', 'author')
ap = dgl.bipartite(transpose(pa_list), 'author', 'ap', 'paper')
pc = dgl.bipartite(pc_list, 'paper', 'pc', 'conf')
cp = dgl.bipartite(transpose(pc_list), 'conf', 'cp', 'paper')
hg = dgl.hetero_from_relations([pa, ap, pc, cp])

features = torch.FloatTensor(author_features)
labels = torch.LongTensor(labels)

print(features.shape)
print(labels.shape)

num_class = 4

alls = [i for i in range(len(a_list))]
train_idx, x, _, _ = train_test_split(alls,labels,test_size=0.2, random_state=52)
eval_idx, test_idx, _,_ = train_test_split(x,_,test_size=0.5,random_state=40)

num_nodes = hg.number_of_nodes('author')
train_mask = get_binary_mask(num_nodes, train_idx)
val_mask = get_binary_mask(num_nodes, eval_idx)
test_mask = get_binary_mask(num_nodes, test_idx)

print(features.shape)
print(train_mask)
print(val_mask)