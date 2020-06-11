import tensorflow as tf
import math


class GATNE:
    def __init__(self, edge_type_count, neighbor_samples, num_nodes,
                 embedding_size, u_num, embedding_u_size, att_head, dim_a, num_sampled, learning_rate
                 ):
        self.edge_type_count = edge_type_count
        self.neighbor_samples = neighbor_samples
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.u_num = u_num
        self.embedding_u_size = embedding_u_size
        self.att_head = att_head
        self.dim_a = dim_a
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate

    def buils_GATNE(self, feature_dic=None, features=None, feature_dim=None):
        with tf.name_scope('input'):
            # Input data
            train_inputs = tf.placeholder(tf.int32, shape=[None])
            train_labels = tf.placeholder(tf.int32, shape=[None, 1])
            train_types = tf.placeholder(tf.int32, shape=[None])
            node_neigh = tf.placeholder(tf.int32, shape=[None, self.edge_type_count, self.neighbor_samples])

        with tf.name_scope('variable'):
            # Parameters to learn
            if feature_dic is not None:
                node_features = tf.Variable(features, name='node_features', trainable=False)
                feature_weights = tf.Variable(tf.truncated_normal([feature_dim, self.embedding_size], stddev=1.0))
                linear = tf.layers.Dense(units=self.embedding_size, activation=tf.nn.tanh, use_bias=True)

                embed_trans = tf.Variable(
                    tf.truncated_normal([feature_dim, self.embedding_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))
                u_embed_trans = tf.Variable(
                    tf.truncated_normal([self.edge_type_count, feature_dim, self.embedding_u_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))

            node_embeddings = tf.Variable(tf.random_uniform([self.num_nodes, self.embedding_size], -1.0, 1.0))
            node_type_embeddings = tf.Variable(
                tf.random_uniform([self.num_nodes, self.u_num, self.embedding_u_size], -1.0, 1.0))
            trans_weights = tf.Variable(
                tf.truncated_normal(
                    [self.edge_type_count, self.embedding_u_size, self.embedding_size // self.att_head],
                    stddev=1.0 / math.sqrt(self.embedding_size)))
            trans_weights_s1 = tf.Variable(
                tf.truncated_normal([self.edge_type_count, self.embedding_u_size, self.dim_a],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            trans_weights_s2 = tf.Variable(
                tf.truncated_normal([self.edge_type_count, self.dim_a, self.att_head],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            nce_weights = tf.Variable(
                tf.truncated_normal([self.num_nodes, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            nce_biases = tf.Variable(tf.zeros([self.num_nodes]))

        with tf.name_scope('embedding'):
            # Look up embeddings for nodes
            if feature_dic is not None:
                node_embed = tf.nn.embedding_lookup(node_features, train_inputs)
                node_embed = tf.matmul(node_embed, embed_trans)
            else:
                node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)

            if feature_dic is not None:
                node_embed_neighbors = tf.nn.embedding_lookup(node_features, node_neigh)
                node_embed_tmp = tf.concat([tf.matmul(
                    tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, 0], [-1, 1, -1, -1]), [-1, feature_dim]),
                    tf.reshape(tf.slice(u_embed_trans, [i, 0, 0], [1, -1, -1]),
                               [feature_dim, self.embedding_u_size])) for i in range(self.edge_type_count)],
                    axis=0)
                node_type_embed = tf.transpose(
                    tf.reduce_mean(tf.reshape(node_embed_tmp,
                                              [self.edge_type_count, -1, self.neighbor_samples, self.embedding_u_size]),
                                   axis=2), perm=[1, 0, 2])
            else:
                node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh)
                node_embed_tmp = tf.concat(
                    [tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1]),
                                [1, -1, self.neighbor_samples, self.embedding_u_size]) for i in
                     range(self.edge_type_count)], axis=0)
                node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1, 0, 2])

        with tf.name_scope('attention'):
            trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
            trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types)
            trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types)

            attention = tf.reshape(tf.nn.softmax(
                tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, self.u_num])),
                [-1, self.att_head, self.u_num])
            node_type_embed = tf.matmul(attention, node_type_embed)
            node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, self.embedding_size])

            if feature_dic is not None:
                node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
                node_embed = node_embed + tf.matmul(node_feat, feature_weights)

            last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)

        with tf.name_scope('metrics'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=last_node_embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.num_nodes))
            plot_loss = tf.summary.scalar("loss", loss)
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return train_inputs, train_labels, train_types, node_neigh, last_node_embed, loss, optimizer
