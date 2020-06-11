import numpy as np

from deepwalk_model import DeepWalk
from deepwalk_utils import load_dataset
from Amazon_evaluation import evaluate_amazon
from Linear_evaluation import evaluate_DBLP

if __name__ == "__main__":

    dataset = 'Amazon'

    walk_length = 10
    num_walks = 20

    embed_size = 200
    window_size = 5
    iters = 100
    min_count = 5
    negative_samples = 5

    G, sentences, word2index, index2word = load_dataset(dataset=dataset, num_walks=num_walks, walk_length=walk_length)

    model = DeepWalk()

    model.train(sentences,
                embed_size=embed_size,
                window_size=window_size,
                iters=iters,
                min_count=min_count,
                negative_samples=negative_samples)
    embeddings = model.get_embeddings(list(word2index.keys()))

    if dataset == 'DBLP':
        evaluate_DBLP(embeddings)

    if dataset == 'Amazon':
        evaluate_amazon(embeddings)