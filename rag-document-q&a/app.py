import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores  import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'RAG Application'

groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most response based on the context
    <context> {context} <context>
    Questions : {input}
"""
)

st.title('RAG Document with Q&A Chatbot with Groq and HuggingFace 😄')

def create_vector_embedding():
    if 'vectors' not in st.session_state:
        # st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('research_papers')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt = st.text_input('Enter your query from the research paper')

if st.button('Document Embedding'):
    create_vector_embedding()
    st.write('Vector Database is ready.')

if user_prompt and "vectors" in st.session_state:

    # 1️⃣ Runnable document chain (replacement of create_stuff_documents_chain)
    document_chain = prompt | llm

    # 2️⃣ Retriever from session_state
    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 4}
    )

    # 3️⃣ Retrieve documents
    start = time.process_time()
    docs = retriever.invoke(user_prompt)

    # 4️⃣ Manually stuff documents into context
    context = "\n\n".join(doc.page_content for doc in docs)

    # 5️⃣ Invoke runnable chain
    response = document_chain.invoke({
        "context": context,
        "input": user_prompt
    })

    print(f"Response time : {time.process_time() - start}")

    # 6️⃣ Show answer
    st.write(response.content)

    # 7️⃣ Show retrieved documents (like old response['context'])
    with st.expander("Document similarity search"):
        for i, doc in enumerate(docs):
            st.write(f"Document {i+1}")
            st.write(doc.page_content)
            st.write("------------")
