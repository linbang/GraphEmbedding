from gensim.models import Word2Vec


class Metapath2vec:
    def __init__(self, sentences, graph):

        self.w2v_model = None
        self._embeddings = {}
        self.sentences = sentences
        self.graph = graph

    def train(self, embed_size=200, window_size=5, iters=10, min_count=5, negative_samples=5):

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