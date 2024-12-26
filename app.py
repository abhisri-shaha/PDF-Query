import streamlit as st
import os
from dotenv import load_dotenv
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from io import StringIO

# Load environment variables
load_dotenv()

# Initialize the connection to Astra DB
cassio.init(
    token=os.environ.get('ASTRA_DB_APPLICATION_TOKEN'),
    database_id=os.environ.get('ASTRA_DB_ID')
)

# Create LangChain embedding and LLM objects for later usage
llm = HuggingFaceHub(repo_id="google/flan-t5-large", huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create LangChain vector store backed by Astra DB
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="pdf_qna",
    session=None,
    keyspace=None,
)

# Text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

def process_pdf(uploaded_file):
    # Read PDF content
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def create_vector_store(raw_text):
    texts = text_splitter.split_text(raw_text)
    astra_vector_store.add_texts(texts[:50])
    return VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Streamlit interface
st.title("PDF Q&A with LangChain and Astra DB")
st.write("Upload your PDF document and ask questions related to it.")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    raw_text = process_pdf(uploaded_file)
    
    # Create the vector store from the document
    astra_vector_index = create_vector_store(raw_text)
    
    # Interaction with the user for asking questions
    query_text = st.text_input("Enter your question:")
    
    if query_text:
        if query_text.lower() == "quit":
            st.stop()
        
        # Get answer from the vector store
        answer = astra_vector_index.query(query_text, llm=llm).strip()
        st.write(f"Answer: {answer}")
