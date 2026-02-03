import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Q&A Chatbot'

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'Hey, you are helpful assistant. Please response me...'),
        ('user', 'Question: {question}')
    ]
)

## Temperature --> Temperature is a parameter that controls the randomness of the model's output. Its value ranges from 0 to 1. A lower temperature (e.g., 0.2) makes the output more focused and deterministic, while a higher temperature (e.g., 0.8) makes it more random and creative.
def generate_response(question, api_key, llm, temperature, max_tokens):
    if api_key.startswith('sk-'):
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens= max_tokens,
            openai_api_key = api_key
        )
    else:
        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens= max_tokens,
            groq_api_key = api_key
        )
    # openai.api_key = api_key
    # llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

## title of the app
st.title('Q&A Chatbot')

## sidebar
st.sidebar.title('Settings')
api_key = st.sidebar.text_input("Enter your api key", type='password')

## model
model_name = st.sidebar.selectbox('Select an Open AI model', ['gpt-4o', 'gpt-4', 'gpt-4-turbo', 'groq:qwen/qwen3-32b', 'llama-3.1-8b-instant'])

## adjust response parameter
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)
max_token = st.sidebar.slider('Max Token ', min_value=100, max_value=1000, value=300)

## main interface
st.write("Ask any questions")
user_input = st.text_input('You : ')

## call model
if user_input:
    response = generate_response(user_input, api_key, model_name, temperature, max_token)
    st.write(response)
else:
    st.write('Please ask some question.....')
