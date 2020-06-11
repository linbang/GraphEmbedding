import time
from numpy import random
import torch
import tensorflow as tf

from gatne_utils import *
from gatne_model import GATNE
from gatne_input import TrainDataInput


def train(network_data, feature_dic=None):
    # vocab是word2index的dict；index2word是一个list；type_nodes是一个list，包含各种edgetype的node[[type1_nodes], [],..]
    index2word, vocab, type_nodes = generate_vocab(network_data)

    edge_types = list(network_data.keys())
    num_nodes = len(index2word)
    edge_type_count = len(edge_types)

    learning_rate = args.learning_rate
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_negative_sampled = args.negative_samples
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples

    # 随机游走，得到训练序列
    g = get_graph(network_data, vocab)  # 构造节点类型相同，边类型不同的图。对应论文第一个部分
    all_walks = []
    for i in range(edge_type_count):
        nodes = torch.LongTensor(type_nodes[i] * args.num_walks)  # 可以理解为每个node采样20个path
        traces, types = dgl.sampling.random_walk(g, nodes,
                                                 metapath=[edge_types[i]] * (neighbor_samples - 1))  # 按照边的类型进行采样
        all_walks.append(traces)

    # 将训练序列拆成训练数据对, 得到邻居节点。
    train_pairs = generate_pairs(all_walks, args.window_size)  # window_size内的都是正样本

    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x]
            iy = vocab[y]
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(
                    list(np.random.choice(neighbors[i][r], size=neighbor_samples - len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))

    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key], :] = np.array(value)

    # 获取model
    model = GATNE(
        edge_type_count=edge_type_count,
        neighbor_samples=neighbor_samples,
        num_nodes=num_nodes,
        embedding_size=embedding_size,
        u_num=u_num,
        embedding_u_size=embedding_u_size,
        att_head=att_head,
        dim_a=dim_a,
        num_sampled=num_negative_sampled,
        learning_rate=learning_rate
    )
    if feature_dic is not None:
        train_inputs, train_labels, train_types, node_neigh, last_node_embed, loss, optimizer = model.buils_GATNE(feature_dic, features, feature_dim)
    else:
        train_inputs, train_labels, train_types, node_neigh, last_node_embed, loss, optimizer = model.buils_GATNE()

    # 训练模型
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        initializer = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(initializer)

        best_score = 0
        patience = 0
        max_step = 5000
        print("start learning")
        for epoch in range(epochs):
            loss_sum_step = 0.0
            loss_sum_epoch = 0.0
            iteration = 0

            d = TrainDataInput(train_pairs, neighbors, batch_size, sess)
            for _, uij in d:
                feed_dict = {train_inputs:uij[0], train_labels:uij[1], train_types:uij[2], node_neigh:uij[3]}
                loss_batch, _, summary = sess.run(
                    [loss, optimizer, merged], feed_dict= feed_dict)

                loss_sum_epoch += loss_batch
                loss_sum_step += loss_batch

                iteration += 1
                if iteration % max_step == 0:
                    print("EPOCH %d step %d , average train loss %f " % (epoch, iteration / max_step, loss_sum_step / max_step))
                    loss_sum_step = 0.0
            print("EPOCH %d, average train loss %f" % (epoch, loss_sum_epoch / iteration))

            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            for i in range(edge_type_count):
                for j in range(num_nodes):
                    final_model[edge_types[i]][index2word[j]] = np.array(
                        sess.run(last_node_embed, {train_inputs: [j], train_types: [i], node_neigh: [neighbors[j]]})[0])

            valid_aucs, valid_f1s, valid_prs, valid_aps = [], [], [], []
            test_aucs, test_f1s, test_prs, test_aps = [], [], [], []
            for i in range(edge_type_count):
                if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    tmp_auc, tmp_f1, tmp_pr, tmp_ap = evaluate(final_model[edge_types[i]],
                                                   valid_true_data_by_edge[edge_types[i]],
                                                   valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)
                    valid_aps.append(tmp_ap)

                    tmp_auc, tmp_f1, tmp_pr, tmp_ap = evaluate(final_model[edge_types[i]],
                                                   testing_true_data_by_edge[edge_types[i]],
                                                   testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
                    test_aps.append(tmp_ap)
            print('EPOCH %d, valid auc: %f' %(epoch, np.mean(valid_aucs)))
            print('EPOCH %d, valid pr: %f' %(epoch, np.mean(valid_prs)))
            print('EPOCH %d, valid f1: %f'%(epoch, np.mean(valid_f1s)))
            print('EPOCH %d, valid ap: %f' % (epoch, np.mean(valid_aps)))

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)
            average_ap = np.mean(test_aps)

            cur_score = np.mean(valid_aucs)
            if cur_score > best_score:
                best_score = cur_score
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print('Early Stopping')
                    break
    return test_aucs[-1], test_f1s[-1], test_prs[-1], test_aps[-1]



if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)

    training_data_by_type = load_training_data(file_name + "/train.txt")  # 训练边集合，按照边类型分
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(  # 用户验证边集合，带label
        file_name + "/valid.txt"
    )
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
        file_name + "/test.txt"
    )
    start = time.time()
    average_auc, average_f1, average_pr, average_ap = train(training_data_by_type)
    end = time.time()

    print("Overall ROC-AUC:", average_auc)
    print("Overall PR-AUC", average_pr)
    print("Overall F1:", average_f1)
    print("Overall AP:", average_ap)
    print("Training Time", end - start)
