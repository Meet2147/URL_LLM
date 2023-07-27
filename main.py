import pickle
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
import os
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def run_question_answering():
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    st.title("LLM For URLS")
    # Set the OpenAI API key
    url = st.sidebar.text_input("Enter URL to load data:", key="url")
    if url:
        os.environ["OPENAI_API_KEY"] = "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"

        # Load data from the provided URL
        loaders = UnstructuredURLLoader(urls=[url])
        data = loaders.load()

        # Split the loaded text into chunks
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        # Create OpenAI embeddings for the documents
        embeddings = OpenAIEmbeddings()

        # Build a FAISS vector store using the documents and embeddings
        vectorStore_openAI = FAISS.from_documents(docs, embeddings)

        # Save the vector store to a file
        with open("faiss_store_openai.pkl", "wb") as f:
            pickle.dump(vectorStore_openAI, f)

        # Load the vector store from the file
        with open("faiss_store_openai.pkl", "rb") as f:
            VectorStore = pickle.load(f)

        # Create an OpenAI instance for question answering
        llm = OpenAI(temperature=0, model_name='text-davinci-003')

        # Create a retrieval question-answering chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

        if "messages" not in st.session_state:
            st.session_state.messages = []

    # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            # Prompt the user for a question or 'exit' to quit
        if query := st.chat_input("How may I help you?"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with get_openai_callback() as cb:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    response = chain(query, return_only_outputs=True)
                    answer = response['answer']
                    source = response['sources']
                    print(response)
                    if response:
                        messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    full_response += f"**Answer:** {answer}\n\n"
                    full_response += f"**Sources:** [{source}]({source})"
                    # full_response += answer
                    
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # else:    # Get the response to the question using the retrieval question-answering chain
    #     response = chain(query, return_only_outputs=True)
    #     st.write(response)

# Streamlit app


if __name__ == "__main__":
    run_question_answering()
