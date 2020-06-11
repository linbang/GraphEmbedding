from gensim.models import Word2Vec
from deepwalk_utils import load_dblp


class DeepWalk:
    def __init__(self):

        self.w2v_model = None
        self._embeddings = {}

    def train(self, sentences, embed_size=200, window_size=5, iters=100, min_count=5, negative_samples=5):

        print("Learning embedding vectors...")
        model = Word2Vec(sentences=sentences,
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

    def get_embeddings(self, nodes):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for node in nodes:
            self._embeddings[node] = self.w2v_model.wv[node]

        return self._embeddings
