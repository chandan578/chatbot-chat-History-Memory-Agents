import streamlit as st
import os
import uuid
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory


# -------------------- ENV --------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# -------------------- STREAMLIT --------------------
st.title("Conversational RAG with PDF + Chat History")
st.write("Upload PDFs and chat with their content")

# -------------------- SESSION INIT --------------------
if "store" not in st.session_state:
    st.session_state.store = {}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------- API KEY --------------------
api_key = st.text_input("Enter your Groq API key", type="password")
if not api_key:
    st.warning("Please enter Groq API key")
    st.stop()

# -------------------- LLM --------------------
llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

session_id = st.text_input("SESSION ID", value="default_session")

# -------------------- EMBEDDINGS --------------------
embeddings = HuggingFaceEmbeddings()

# -------------------- PDF UPLOAD --------------------
upload_files = st.file_uploader(
    "Upload PDF file(s)",
    type="pdf",
    accept_multiple_files=True
)

if upload_files and st.session_state.vectorstore is None:
    documents = []

    for upload_file in upload_files:
        temp_path = f"./temp_{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(upload_file.getvalue())

        loader = PyPDFLoader(temp_path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    splits = splitter.split_documents(documents)

    st.session_state.vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    st.success("PDF indexed successfully!")

# -------------------- PROMPTS --------------------
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question, "
     "rewrite the question to be standalone if needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following context to answer the question. "
    "If you don't know the answer, say you don't know.\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# -------------------- CHAINS --------------------
standalone_question_chain = (
    contextualize_prompt
    | llm
    | StrOutputParser()
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever()

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=standalone_question_chain
        )
        .assign(
            context=lambda x: retriever.invoke(x["standalone_question"])
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # -------------------- CHAT INPUT --------------------
    user_input = st.text_input("Ask a question about the PDF")

    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.success(response)
        st.write("Chat History:")
        st.write(st.session_state.store[session_id].messages)

else:
    st.info("Upload a PDF to start chatting.")
