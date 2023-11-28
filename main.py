import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import pandas as pd
import openai
import pdfplumber
import requests
from io import BytesIO

# Set up OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel, or PDF", type=["csv", "xlsx", "pdf"])
url_input = st.sidebar.text_input("Or enter a URL")

data = None

try:
    if uploaded_file:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            data = df[df.columns[0]].tolist()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            data = df[df.columns[0]].tolist()
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                data = [page.extract_text() for page in pdf.pages if page.extract_text() is not None]

    elif url_input:
        response = requests.get(url_input)
        data = response.text

    if data:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
            retriever=vectorstore.as_retriever())

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Chat interface
        container = st.container()
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here :)", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = chain({"question": user_input, "chat_history": st.session_state['history']})
                st.session_state['history'].append((user_input, output["answer"]))

        response_container = st.container()
        with response_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state['history']):
                message(user_msg, is_user=True, key=f'user_{i}', avatar_style="big-smile")
                message(bot_msg, key=f'bot_{i}', avatar_style="thumbs")

except Exception as e:
    st.error(f"An error occurred: {e}")
