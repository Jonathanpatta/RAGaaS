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

load_dotenv(".env")

st = time.time()

embedder = Embedder()
vsm = VectorStoreManager()

retriever = CustomQdrantRetriever(documents=[],k=3,embedder=embedder,vsm=vsm)
llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")



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
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# res = chain.invoke({"input": "what is the history of the Spinning jenny"})

# print(time.time()-st)

# print(res['answer'])
# print(res['context'])

st = time.time()

res = chain.invoke({"input": "Richard Henry 'Peter' Sellers birth and death date?"})

print(time.time()-st)


# print(res['answer'])
print(res['context'])

