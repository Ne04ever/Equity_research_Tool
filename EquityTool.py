import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Cohere

load_dotenv()
llm = Cohere(max_tokens=256, temperature=0.75)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

file_path = "vector_index.pkl"

# Title should appear before the query input
st.title("Equity Research Tool")
st.sidebar.title("News Articles URL")

button = st.sidebar.button('Process URLs')
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

placeholder = st.empty()

if button:
    loaders = UnstructuredURLLoader(urls=urls)
    placeholder.text('Data loading ........')
    data = loaders.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    placeholder.text('Text splitting ........')

    # Split documents into chunks
    docs = text_splitter.split_documents(data)
    vectorindex_hug = FAISS.from_documents(docs, embeddings)
    placeholder.text('Vector embedding started ........')
    time.sleep(2)

    # Save the vector index
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_hug, f)

# Now show the query input and the title will be above it
query = placeholder.text_input('Question: ')
if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.subheader(result['answer'])
