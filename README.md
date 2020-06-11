# GraphEmbedding
Implement of several Graph Embedding methods，including Homogeneous and Heterougeneous Graph.

## Methods

| Method       | Paper                                                        |
| ------------ | ------------------------------------------------------------ |
| Deepwalk     | [DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf) |
| LINE         | [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf) |
| Node2vec     | [node2vec: Scalable Feature Learning for Networks](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf) |
| GCN          | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) |
| GAT          | [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) |
| Metapath2vec | [Metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) |
| HAN          | [Heterogeneous Graph Attention Network](https://arxiv.org/pdf/1903.07293.pdf) |
| GATNE        | [Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/pdf/1905.01669.pdf) |

## Environment

```python
python
networkx
dgl
pytorch
tensorflow
```

## Input

### Supervised learning

We use DBLP dataset and set CrossEntropy loss. 

We get node embeddings from model and use LR as classifier.

![image-20200611131417986](https://i.loli.net/2020/06/11/Nvgq8XSJaz5wIQR.png)

### Un-Supervised learning

We use Amazon dataset to realize link prediction task, using NCE loss.

![image-20200611131706554](https://i.loli.net/2020/06/11/eYTvNtHdbGxrZqy.png)