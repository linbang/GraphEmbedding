import tensorflow as tf
import numpy as np


class TrainDataInput:
    def __init__(self, pairs, neighbors, batch_size, sess):
        self.batch_size = batch_size
        self.pairs = pairs
        self.neighbors = neighbors
        self.epoch_size = len(self.pairs) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.pairs):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size - 1:
            raise StopIteration

        x, y, t, neigh = [], [], [], []
        for j in range(self.batch_size):
            index = self.i + j
            x.append(self.pairs[index][0])
            y.append(self.pairs[index][1])
            t.append(self.pairs[index][2])
            neigh.append(self.neighbors[self.pairs[index][0]])

        self.i += 1

        return 0, (np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32),
               np.array(neigh).astype(np.int32))