

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import time



def batch(list1, size):

    list_of_lists = []
    for i in range(0, len(list1), size):
        list_of_lists.append(list1[i:i + size])
    return list_of_lists

class Embedder():

    def __init__(self) -> None:
        self.model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

    def embed(self,chunks):

        # st = time.time()
        batched = batch(chunks,5)

        embeddings = []

        for sen in batched:
            embeddings += self.model.encode(sen).tolist()
        
        # print("time taken by embedding: ",time.time()-st)
        return embeddings



    def embed_sentence(self,sentence):

        st = time.time()
        
        embeddings = self.model.encode(sentences=[sentence])
        embedding_list = embeddings[0].tolist()
        print("time taken by embedding: ",time.time()-st)
        return embedding_list

    def store_embeddings(embeddings):
        pass

    def openai_embed(chunks):
        pass



# sentences = ['That is a happy person', 'That is a very happy person']

# embedder = Embedder()
# embeddings = embedder.embed(sentences)
# print(cos_sim(embeddings[0], embeddings[1]))
# print(embeddings.__sizeof__())
# print(embeddings[0].tolist())