import pickle
import dgl
import torch


def transpose(l):
    new_l = []
    for k in l:
        new_l.append(tuple([k[1], k[0]]))
    return new_l


def load_dblp(num_walks, metapaths):
    with open('../dataset/DBLP/output/DBLP_Metapath2vec.pickle', 'rb') as f:
        a_list, p_list, c_list, node_list = pickle.load(f)
        pa_list, pc_list = pickle.load(f)

    print(len(pa_list))
    print(len(pc_list))

    author_ids = [node_list.index(i) for i in a_list]

    # 构造异构网络
    pa = dgl.bipartite(pa_list, 'paper', 'pa', 'author')
    ap = dgl.bipartite(transpose(pa_list), 'author', 'ap', 'paper')
    pc = dgl.bipartite(pc_list, 'paper', 'pc', 'conf')
    cp = dgl.bipartite(transpose(pc_list), 'conf', 'cp', 'paper')
    hg = dgl.hetero_from_relations([pa, ap, pc, cp])

    # 随机游走
    sentences = []
    for metapath in metapaths:
        traces, types = dgl.sampling.random_walk(hg, author_ids * num_walks, metapath=metapath)
        for s in traces.tolist():
            sentences.append([node_list[i] for i in s])

    return hg, sentences, node_list


def generate_vocab(network_data):
    nodes, index2word = [], []
    for edge_type in network_data:
        node1, node2 = zip(*network_data[edge_type])
        index2word = index2word + list(node1) + list(node2)

    index2word = list(set(index2word))
    vocab = {}
    for index, word in enumerate(index2word):
        vocab[word] = index

    for edge_type in network_data:
        node1, node2 = zip(*network_data[edge_type])
        tmp_nodes = list(set(list(node1) + list(node2)))
        tmp_nodes = [vocab[word] for word in tmp_nodes]
        nodes.append(tmp_nodes)

    return index2word, vocab, nodes


def get_graph(network_data, vocab):
    graphs = []
    num_nodes = len(vocab)

    for edge_type in network_data:
        tmp_data = network_data[edge_type]
        edges = []
        for edge in tmp_data:
            edges.append((vocab[edge[0]], vocab[edge[1]]))
            edges.append((vocab[edge[1]], vocab[edge[0]]))
        g = dgl.graph(edges, etype=edge_type, num_nodes=num_nodes)
        graphs.append(g)
    graph = dgl.hetero_from_relations(graphs)

    return graph


def load_amazon(num_walks):
    trainfile = '../dataset/Amazon/data/train.txt'
    node_list = set()
    edge_data_by_type = dict()
    for line in open(trainfile, 'r').readlines():
        line = line.strip().split(' ')
        if line[0] not in edge_data_by_type:
            edge_data_by_type[line[0]] = list()
        x, y = str('I'+line[1]), str('I'+line[2])
        edge_data_by_type[line[0]].append((x, y))
        edge_data_by_type[line[0]].append((x, y))
        node_list.add(x)
        node_list.add(y)

    node_list = list(node_list)
    print("all node number is:", len(node_list))

    index2word, vocab, type_nodes = generate_vocab(edge_data_by_type)
    edge_types = list(edge_data_by_type.keys())

    g = get_graph(edge_data_by_type, vocab)
    all_walks = []
    for i in range(len(edge_data_by_type.keys())):
        nodes = torch.LongTensor(type_nodes[i] * num_walks)  # 可以理解为每个node采样20个path
        traces, types = dgl.sampling.random_walk(g, nodes, metapath=[edge_types[i]] * 9)  # 按照边的类型进行采样
        for s in traces.tolist():
            all_walks.append([index2word[i] for i in s])

    return g, all_walks, index2word


def load_dataset(dataset, num_walks, metapaths=None):
    if dataset == 'DBLP':
        return load_dblp(num_walks, metapaths)
    if dataset == 'Amazon':
        return load_amazon(num_walks)

#g, all_walks, index2word = load_amazon(1)
#print(all_walks)
#print(index2word)