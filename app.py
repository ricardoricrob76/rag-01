import os
import fitz  # PyMuPDF
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ChatPDF com RAG", layout="wide")
st.title("📄 Chat com PDF usando RAG e IA Generativa")

# === 1. Upload do PDF ===
pdf_file = st.file_uploader("Envie um PDF", type=["pdf"])

if pdf_file:
    with open(f"docs/{pdf_file.name}", "wb") as f:
        f.write(pdf_file.read())

    # === 2. Leitura e divisão dos textos ===
    loader = PyMuPDFLoader(f"docs/{pdf_file.name}")
    docs = loader.load()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    # === 3. Embeddings e banco vetorial ===
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # === 4. Criação do Chain RAG ===
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("PDF processado com sucesso! Agora você pode fazer perguntas.")

    # === 5. Interface de perguntas ===
    user_question = st.text_input("Pergunte algo sobre o documento:")

    if user_question:
        with st.spinner("Consultando..."):
            resposta = rag_chain.run(user_question)
        st.markdown(f"**Resposta:** {resposta}")
