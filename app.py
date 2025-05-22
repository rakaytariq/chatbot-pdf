import streamlit as st
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load models once
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# PDF Text Extraction
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Chunk and Embed
def chunk_and_embed(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embedded_chunks = embedder.encode(chunks)
    return chunks, embedded_chunks

# Create FAISS index
def create_faiss_index(embedded_chunks):
    dimension = len(embedded_chunks[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embedded_chunks))
    return index

# Answer questions
def generate_response(query, chunks, embedded_chunks, index):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = " ".join(retrieved_chunks)
    result = qa_pipeline(question=query, context=context)
    return result['answer']

# Streamlit UI
st.title("📄 PDF Chatbot (Free & Offline)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks, embedded_chunks = chunk_and_embed(text)
    index = create_faiss_index(embedded_chunks)
    st.success("✅ PDF processed successfully!")

    query = st.text_input("💬 Ask a question about the document:")
    if query:
        response = generate_response(query, chunks, embedded_chunks, index)
        st.write(f"🧠 Answer: {response}")
