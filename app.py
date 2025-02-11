from langchain.llms import HuggingFaceHub
import os
import torch
import streamlit

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory  # Import this!
from langchain_ollama import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
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

from huggingface_hub import InferenceClient

client = InferenceClient(
	provider="hf-inference",
	api_key=st.secrets.huggingfacetoken
)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets.huggingfacetoken

# 2. Load Data & Setup Vectorstore (Simplified for now)
@st.cache_resource
def store_data(_data, _embeddings):
  # vector store
  db = FAISS.from_documents(data, embeddings)
  return db

embeddings = embed(data, 'cpu', 'sentence-transformers/all-MiniLM-L6-v2')
db = store_data(data, embeddings)

# 3. Create Chat Model
@st.cache_resource  # Cache this function to load the model only once
def initialize_model():
    llm = HuggingFaceHub(
    repo_id="microsoft/Phi-3-mini-4k-instruct", 
    model_kwargs={"temperature": 0.5})
    return llm
# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history
if "db" not in st.session_state:
    st.session_state.db = db
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None


if not st.session_state.model:
    with st.spinner("Initializing Chatbot..."):
        st.session_state.model = initialize_model()


from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain.memory import ConversationBufferMemory


def generate_response(question):
    db = st.session_state.db
    model = st.session_state.model
    # perform a similarity search and retrieve the context from our documents
    results = db.similarity_search(question, k=3)
        # join all context information into one string
    context = "n".join([document.page_content for document in results])

         # chatbot initial prompt template

	template = """ "role": "system", "content":You are WiChat, the chatbot for the Worldbank Ideas Project. You are friendly and follow instructions to answer questions extremely well. Please be truthful and give direct answers. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the response short and concise in at most five sentences. If the user chats in a different language, translate accurately and respond in the same language. You will provide specific details and accurate answers to user queries on the Worldbank Ideas Project.
                   Use the following pieces of context to answer the user's question.
                   Context: {context}
		   "role": "user", "content":{question}
                   "role": "assistant", "content": """
    prompt = PromptTemplate(input_variables=["context",  "question"], template=template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    retrievalqa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory = memory,
        chain_type_kwargs={"prompt": prompt}
    )
    answer = retrievalqa({"query": question})
	
    return answer["result"]

# 8. Streamlit UI
st.title("WiChat")
st.markdown("Welcome! I'm here to answer your questions on the World Bank IDEAS Project.")

# 9. Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"  # Or generate a unique ID


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
