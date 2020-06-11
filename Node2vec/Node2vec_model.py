from gensim.models import Word2Vec
from Node2vec_utils import RandomWalker

class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1, use_rejection_sampling=0):

        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(
            graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=200, window_size=5, iters=100, min_count=5, negative_samples=5):

        print("Learning embedding vectors...")
        model = Word2Vec(sentences=self.sentences,
                         size=embed_size,
                         window=window_size,
                         iter=iters,
                         min_count=min_count,
                         sg=1,
                         hs=0,
                         negative=negative_samples)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self, node_list):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for index,word in enumerate(node_list):
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings