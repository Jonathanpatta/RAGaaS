from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


import time

from embed import Embedder
from langchain_openai import OpenAIEmbeddings

import os

class VectorStoreManager:
   def __init__(self) -> None:
      self.client = QdrantClient(
         url=os.getenv("QDRANT_CLOUD_URL"), 
         # url=os.getenv("QDRANT_LOCAL_URL"),
         api_key=os.getenv("QDRANT_API_KEY"),
      )

   
   def get_vectors(self,query,embedder,k=1):
      
      vect = embedder.embed_sentence(query)
      search_results = self.client.search(
         collection_name="test",
         query_vector=vect,
         limit=k,
         with_payload=True,
      )
      ids = []
      
      res_docs = []
      # print(search_results)
      for res in search_results:
         ids.append(int(res.payload['id']))

         # print(type(res.payload['meta_data']))

         d = Document(page_content=res.payload['data_pt'])
         # d.metadata
         d.metadata = res.payload['meta_data']
         res_docs.append(d)
      
      return res_docs
      


# qdrant_client = 

# qdrant_client.recreate_collection(
#    collection_name="test",
#    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
# )

# qdrant_client.recreate_collection(
#    collection_name="pdf-test",
#    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
# )




def batch(list1, size):

      list_of_lists = []
      for i in range(0, len(list1), size):
         list_of_lists.append(list1[i:i + size])
      return list_of_lists

# def load_embeddings_from_file_and_upload():
#    docs = get_docs_from_text("testdata/archive/AllCombined.txt")
#    import pickle
#    with open('testdata/wiki-embeddings/embeddings720.pkl','rb') as f:
#       embeddings = pickle.load(f)

#    print(len(embeddings))
#    t = time.time()
#    points = []
#    for idx, vector in enumerate(embeddings):
#       p = PointStruct(
#                id=idx,
#                vector=embeddings[idx],
#                payload={"id": str(idx),
#                   "data_pt": str(docs[idx].page_content),
#                   "meta_data": docs[idx].metadata
#                   }
#          )
#       points.append(p)

   

#    batched_points = batch(points,1000)

#    for id,batch_pts in enumerate(batched_points):
#       qdrant_client.upsert(
#          collection_name="test",
#          points=batch_pts
         
#       )
#       print(time.time()-t)



# def load_pdf_embeddings_from_file_and_upload():

#    docs = get_docs_from_pdf("testdata/krr_report3.pdf")
   
#    embeddings = embed(docs_to_sentences(docs))

#    print(len(embeddings))
#    t = time.time()
#    points = []
#    for idx, vector in enumerate(embeddings):
#       p = PointStruct(
#                id=idx,
#                vector=embeddings[idx],
#                payload={
#                   "id": str(idx),
#                   "data_pt": str(docs[idx].page_content),
#                   "meta_data": docs[idx].metadata,
#                         }
#          )
#       points.append(p)

   

#    batched_points = batch(points,1000)

#    for id,batch_pts in enumerate(batched_points):
#       qdrant_client.upsert(
#          collection_name="pdf-test",
#          points=batch_pts
         
#       )
#       print(time.time()-t)


# load_embeddings_from_file_and_upload()

# load_pdf_embeddings_from_file_and_upload()

