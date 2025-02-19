# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
#move to the folder where the scraped data is on your device
#from google.colab import drive
#drive.mount('/content/drive')
# %cd /content/drive/MyDrive/Data/WiChat

#!pip install virtualenv
#!virtualenv /content/drive/MyDrive/Data/WiChat
#!source /content/drive/MyDrive/Data/WiChat/bin/activate

#!curl -fsSL https://ollama.com/install.sh | sh

# Commented out IPython magic to ensure Python compatibility.
#!pip install colab-xterm
# %load_ext colabxterm

# Commented out IPython magic to ensure Python compatibility.
# %xterm
#pip install requirements.txt
#import os
#os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'

# install dependencies
#!pip install -qU langchain
#!pip install -qU transformers
#!pip install -qU langchain-community
#!pip install -qU unstructured
#!pip install -qU sentence-transformers
#!pip install -qU faiss-cpu
#!pip install -qU langchain_ollama
#!pip install -qU streamlit
import ollama
from ollama import chat
import streamlit as st
from transformers import AutoTokenizer, pipeline  # Import necessary transformers modules


# --- Langchain Imports ---
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

import re
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ensure the notebook is in the same folder as the data files
# load all txt files
def load_data(path):
  loader1 = DirectoryLoader(path, glob = '*.txt', show_progress = True)
  # get content of txt files
  docs = loader1.load()

  return docs


def get_chunks(docs):

    # split the txt files into chunks of 1000 characters and 150 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
    chunks = text_splitter.split_documents(docs)


    return chunks

path = 'scraped_data'
docs = load_data(path)

data = get_chunks(docs)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# embed data sources
def embed(data, device, model):
  model_kwargs = {'device': device}
  encode_kwargs = {'normalize_embeddings': False}

  embeddings = HuggingFaceEmbeddings(
    model_name = model,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
  )
  return embeddings

def store_data(data, embeddings):
  # vector store
  db = FAISS.from_documents(data, embeddings)
  return db

embeddings = embed(data, 'cpu', 'sentence-transformers/all-MiniLM-L6-v2')
db = store_data(data, embeddings)

# --- Application Title and Description ---
st.title("WiChat")
st.markdown("Welcome!!! I'm WiChat. What can I help you with today?")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history
if "db" not in st.session_state:
    st.session_state.db = db
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None


# --- Initialize Chatbot Model (Load on App Start) ---
@st.cache_resource  # Cache this function to load the model only once
def initialize_model():
    model = "phi3:mini"
    ollama.pull(model)
    model = ollama.chat(model)
    return model

if not st.session_state.model:
    with st.spinner("Initializing Chatbot..."):
        st.session_state.model = initialize_model()



# Define the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are WiChat, the chatbot for the Worldbank Ideas Project. You are friendly and follow instructions to answer questions extremely well. Please be truthful and give direct answers. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the response short and concise in at most five sentences. If the user chats in a different language, translate accurately and respond in the same language. You will provide specific details and accurate answers to user queries on the Worldbank Ideas Project."),
         MessagesPlaceholder("chat_history"),
        ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question.")
    ]
)

# --- Create RetrievalQA chain ---
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
    
st.session_state.messages = store

retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

history_aware_retriever = create_history_aware_retriever(st.session_state.model, retriever, contextualize_q_prompt)

question_answer_chain = create_stuff_documents_chain(st.session_state.model, prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,   
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- Response Generation ---
def generate_response(query):
    
    return conversational_rag_chain.invoke({"input": query}, config={"configurable": {"session_id": "1"}})["answer"]
     

# --- User Input ---
user_input = st.chat_input("Ask WiChat anything...")

# --- Process User Input ---
if user_input:

    # --- Generate Chatbot Response ---
    with st.spinner():
        #chatbot_response = st.session_state.model(user_input)
        chatbot_response = generate_response(user_input)

        # --- Append chatbot message to chat history ---
        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Sidebar (Menu and Account Options) ---
with st.sidebar:
    st.header("Options")

    # --- Menu Options ---
    if st.button("Toggle Theme"):
        # Implement Theme Toggle Logic (Streamlit has built-in theme support)
        st.info("Theme toggle functionality to be implemented.")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    if st.button("Help"):
        st.info("Help information to be implemented.")

    # --- Account Options ---
    st.subheader("Account")

    login_button = st.button("Login")  # Direct navigation, no popup
    signup_button = st.button("Sign Up") # Direct Navigation, no popup
