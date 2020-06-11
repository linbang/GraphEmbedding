from Node2vec_model import Node2Vec
from Node2vec_utils import load_dataset
from Load import load_dblp_labels
from Linear_evaluation import evaluate_DBLP
from Amazon_evaluation import evaluate_amazon

if __name__ == "__main__":

    dataset = 'Amazon'

    embed_size = 200
    window_size = 5
    iters = 1
    min_count = 5
    negative_samples = 5

    g, node_list = load_dataset(dataset)
    model = Node2Vec(g,
                     walk_length=10,
                     num_walks=20,
                     p=0.25,
                     q=4,
                     workers=1,
                     use_rejection_sampling=0)

    model.train(embed_size=embed_size, window_size=window_size, iters=iters, min_count=min_count,
                negative_samples=negative_samples)
    embeddings = model.get_embeddings(node_list)

    if dataset == 'DBLP':
        evaluate_DBLP(embeddings)

    if dataset == 'Amazon':
        evaluate_amazon(embeddings)
