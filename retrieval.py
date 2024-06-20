# from langchain_community.vectorstores import Qdrant
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
# from embed import embed_sentence

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from embed import Embedder
# from chunking import get_docs_from_text,get_docs_from_pdf
from qdrant import VectorStoreManager
import time



class CustomQdrantRetriever(BaseRetriever):

    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    embedder: Embedder
    """embedder"""

    vsm: VectorStoreManager
    """Vector Store Manager"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        st = time.time()
        res = self.vsm.get_vectors(query,self.embedder,k=3)
        
        
        # for i in ids:
        #     res_docs.append(pdf_docs[i])
        
        # print(res_docs)

        print("time taken for retrieval:",time.time()-st)
        return res



# embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-large-en-v1.5",model_kwargs={"trust_remote_code":True})
# db = Qdrant(client=qdrant_client,collection_name="test",embeddings=embeddings)
# retriever = db.as_retriever()

# import time
# print("hi")

# start = time.time()
# retriever = CustomQdrantRetriever(documents=[],k=1)
# docs = retriever.invoke("Stephen Hawking")

# print(time.time()-start)

# print(docs)