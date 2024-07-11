from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from retrieval import CustomQdrantRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from embed import Embedder
from qdrant import VectorStoreManager

import time
import os

from dotenv import load_dotenv


class ChainQuerier:

    def __init__(self,embedder,vsm):

        

        st = time.time()

        # self.embedder = Embedder()
        self.embedder = embedder
        # self.vsm = VectorStoreManager()
        self.vsm = vsm

        self.retriever = CustomQdrantRetriever(documents=[],k=3,embedder=self.embedder,vsm=self.vsm)
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")

        print("Chain Querier Init complete:", time.time()-st)



    def query(self,input):
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        chain = create_retrieval_chain(self.retriever, question_answer_chain)

        st = time.time()

        res = chain.invoke({"input": input})

        res['time_taken_for_query'] = time.time()-st

        # print(time.time()-st)

        return res
        # print(res['context'])


load_dotenv(".env")

embedder = Embedder()
vsm1 = VectorStoreManager()

cq = ChainQuerier(embedder=embedder,vsm=vsm1)
res = cq.query("Richard Henry 'Peter' Sellers birth and death date?")
print(res['answer'])