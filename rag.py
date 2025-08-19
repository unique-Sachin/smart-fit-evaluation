from langchain_openai import ChatOpenAI,OpenAIEmbeddings,OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from uuid import uuid4
import getpass
import chromadb
import os
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

loader = CSVLoader(file_path="/Users/sachinmishra/Documents/Evaluation/nutrition.csv")

data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(documents=data)



embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="nutrition_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)


client = chromadb.PersistentClient(path="./chroma_langchain_db")
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)


retriever = vector_store.as_retriever(k=4)

# docs = retriever.invoke("Lemonade")



template = """You are an nutrition assistant for question-answering tasks, Use exact information from context and show numbers in answer.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)



rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | LLM
    | StrOutputParser()
)


chat_history = []

def query_about_nutrition(query:str) -> str:
    resp = rag_chain.invoke(query)
    history = {"query":query,"res":resp}
    chat_history.append(history)
    return resp

