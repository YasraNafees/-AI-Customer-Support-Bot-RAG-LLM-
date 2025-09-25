import os
import shutil
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 1. Load dataset
def load_data(file_path="my_data.csv"):
    df = pd.read_csv(file_path, encoding="latin1")
    docs = []
    for _, row in df.iterrows():
        text = f"""
        Ticket Type: {row['Ticket Type']}
        Subject: {row['Ticket Subject']}
        Description: {row['Ticket Description']}
        Resolution: {row['Resolution']}
        Priority: {row['Ticket Priority']}
        Satisfaction: {row['Customer Satisfaction Rating']}
        """
        docs.append(Document(page_content=text.strip()))
    return docs

# 2. Create or load vectorstore
def create_vectorstore(docs=None):
    persist_dir = "vectorstore"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_dir) and docs is None:
        # Load existing
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        # Rebuild if docs are given
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
        vectordb.persist()
        return vectordb

# 3. Load LLM
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
# 4. Create QA Chain
def get_qa(vectordb):
    llm = get_llm()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # retrieve 3 docs max
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa
