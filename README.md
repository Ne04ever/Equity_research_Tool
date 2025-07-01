# 📈 Equity Research Tool with LangChain + FAISS + Streamlit
This Equity Research Tool enables analysts and users to input URLs of financial or news articles and ask questions about their content using natural language. It processes unstructured web content, splits it into chunks, embeds it, and builds a searchable knowledge base to answer queries with citations.

## 🚀 Features
* 🔗 Input URLs of financial articles
* 🧠 Extracts and chunks unstructured text
* 💡 Embeds using HuggingFace sentence transformer
* 🔎 Retrieves contextually relevant content using FAISS
* 🗣️ Queries answered using Cohere LLM
* 💬 Provides source-aware answers
* 🌐 Interactive Streamlit web interface

## 🔎 How It Works
* User enters 1–3 article URLs via the sidebar.
* Text is extracted and chunked using LangChain's RecursiveCharacterTextSplitter.
* Text chunks are embedded using HuggingFace (all-mpnet-base-v2) and stored in a FAISS index.
* User enters a question related to the content.
* The app uses a retriever + LLM QA chain to:
    * Find the most relevant chunks
    * Pass them to the Cohere LLM
    * Return a natural language answer with source(s)

