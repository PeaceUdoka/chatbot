import streamlit as st
import openai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory  # Import this!

# 1. Authentication
openai.api_key = st.secrets["OPENAI_API_KEY"]
if not openai.api_key:
    st.warning("OpenAI API key not found. Please add it to streamlit secrets.")
    st.stop()

MODEL = "gpt-4o"  # Specify the GPT-4o model

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


# 2. Load Data & Setup Vectorstore (Simplified for now)
@st.cache_resource
def store_data(data, embeddings):
  # vector store
  db = FAISS.from_documents(data, embeddings)
  return db

embeddings = embed(data, 'cpu', 'sentence-transformers/all-MiniLM-L6-v2')
db = store_data(data, embeddings)

# 3. Create Chat Model
llm = ChatOpenAI(model_name=MODEL, openai_api_key=openai.api_key, temperature=0.7)

# 4. Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are WiChat, the chatbot for the Worldbank Ideas Project. Be truthful, friendly and give direct answers. If you don't know the answer, just say that you don't know, don't try to make things up."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# 5. Contextualize prompt and load history
contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 6. Statefully manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 7. RAG Chain construction
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 8. Streamlit UI
st.title("WiChat")
st.markdown("Welcome! Ask me anything related to the World Bank.")

# 9. Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"  # Or generate a unique ID

# 10. Chat Input
if prompt := st.chat_input(placeholder="Ask me anything!"):
    st.session_state.prompt = prompt  #Save input text
    st.chat_message("user").write(prompt)
    st.session_state.past_messages = get_session_history(st.session_state.session_id).messages # load messages with custom function
    with st.spinner():
        response = conversational_rag_chain.invoke(
            {"input": st.session_state.prompt},
            config={"configurable": {"session_id": "1"}},
        )

    #11 Get assistant history and print
    st.chat_message("assistant").write(response["answer"])
    streamlit_chat_history = get_session_history(st.session_state.session_id)

#12. Print all messages at the end:
if "session_id" in st.session_state:
    messages = get_session_history(st.session_state.session_id).messages
    for msg in messages:
        st.chat_message(msg.type).write(msg.content)
