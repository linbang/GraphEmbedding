
from LINE_model import LINE
from LINE_utils import load_dataset

from Linear_evaluation import evaluate_DBLP
from Amazon_evaluation import evaluate_amazon

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    dataset = 'Amazon'
    batch_size = 1024
    epochs = 50

    g, node_list = load_dataset(dataset)

    model = LINE(g, embedding_size=200, order='second')
    model.train(batch_size=batch_size, epochs=epochs, verbose=2)
    embeddings = model.get_embeddings(node_list)

    if dataset == 'DBLP':
        evaluate_DBLP(embeddings)
    if dataset == 'Amazon':
        evaluate_amazon(embeddings)

    #plot_embeddings(embeddings)