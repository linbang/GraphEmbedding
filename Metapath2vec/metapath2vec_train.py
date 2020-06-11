
from metapath2vec_model import Metapath2vec
from metapath2vec_utils import load_dataset

from Linear_evaluation import evaluate_DBLP
from Amazon_evaluation import evaluate_amazon

if __name__ == "__main__":

    dataset = 'Amazon'

    if dataset == 'DBLP':
        metapaths = [['ap','pa']*3, ['ap','pc','cp','pa']*2]
        num_walk = 10
        graph, sentences, node_list = load_dataset(dataset, num_walk, metapaths)
        model = Metapath2vec(sentences, graph)
        model.train()
        embeddings = model.get_embeddings(node_list)
        evaluate_DBLP(embeddings)
    if dataset == 'Amazon':
        num_walk = 10
        graph, sentences, index2word = load_dataset(dataset, num_walk)
        model = Metapath2vec(sentences, graph)
        model.train()
        embeddings = model.get_embeddings(index2word)
        evaluate_amazon(embeddings)
