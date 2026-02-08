import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper
)
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔎Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password"
)

# ----------------------------
# Tools
# ----------------------------
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=1000
    )
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=1000
    )
)

search = DuckDuckGoSearchRun()

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi 👋 I can search the web, Wikipedia, and Arxiv. What do you want to know?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------------------
# Chat Input
# ----------------------------
prompt = st.chat_input("Ask something…")

if prompt and api_key:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    # Run tools manually
    search_result = search.run(prompt)
    wiki_result = wiki.run(prompt)
    arxiv_result = arxiv.run(prompt)

    final_prompt = f"""
You are a helpful AI assistant.
Answer the user's question using the information below.

Web Search:
{search_result}

Wikipedia:
{wiki_result}

Arxiv:
{arxiv_result}

User Question:
{prompt}
"""

    response = llm.invoke(final_prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )

    st.chat_message("assistant").write(response.content)
