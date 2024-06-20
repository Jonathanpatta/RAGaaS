from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader,TextLoader
from langchain_community.document_loaders import CSVLoader
from embed import embed




def get_docs_from_text(path):
    loader = TextLoader(path,encoding="utf-8")
    loaded_documents = loader.load()
    # print(loaded_documents)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.split_documents(loaded_documents)
    # print(len(docs))

    return docs




def get_docs_from_pdf(path):
    loader = UnstructuredPDFLoader(path)
    loaded_documents = loader.load()
    # print(loaded_documents)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.split_documents(loaded_documents)
    # print(len(docs))

    return docs


def docs_to_sentences(docs):
    sentences = []
    for doc in docs:
        sentences.append(doc.page_content)
    
    return sentences


# docs = get_docs_from_pdf("testdata/krr_report3.pdf")
# docs = get_docs_from_text("testdata/archive/2of2/wiki_00")
# docs = docs[:200]
# print(len(docs))
# embed(docs_to_sentences(docs))

# print(docs[2])