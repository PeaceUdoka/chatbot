import streamlit as st
import openai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Authentication
openai.api_key = st.secrets["OPENAI_API_KEY"]
if not openai.api_key:
    st.warning("OpenAI API key not found. Please add it to streamlit secrets.")
    st.stop()

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


# 2. Load Data & setup vectorstore

def store_data(data, embeddings):
  # vector store
  db = FAISS.from_documents(data, embeddings)
  return db

embeddings = embed(data, 'cpu', 'sentence-transformers/all-MiniLM-L6-v2')
db = store_data(data, embeddings)

# 3. Create Chat Model
MODEL = "gpt-4o"  # Specify the model
llm = ChatOpenAI(model_name=MODEL, openai_api_key=openai.api_key, temperature=0.7)

# 4. Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are WiChat, the chatbot for the Worldbank Ideas Project. Use only the context to answer the question. Be truthful and give direct answers. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the response short and concise in at most five sentences. If the user chats in a different language, translate accurately and respond in the same language. You will provide specific details and accurate answers to user queries on the Worldbank Ideas Project."),
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

# 6. Create history retriever
history_aware_retriever = create_history_aware_retriever(llm, db.as_retriever(), contextualize_q_prompt)

# 7. Create chain for answering questions
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 8. Create RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 9. Setup conversational rag chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: StreamlitChatMessageHistory(key="chat_history"),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 10. Streamlit app layout
st.title("WiChat")
st.markdown("Welcome!!! I'm WiChat. What can I help you with today?")

# Initialize Streamlit chat history
streamlit_chat_history = StreamlitChatMessageHistory(key="chat_history")

# Show previous chat messages
messages = streamlit_chat_history.messages
for msg in messages:
    st.chat_message(msg.type).write(msg.content)

# Get answer to user question
if prompt_text := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(prompt_text)

    # 11. Invoke the chain, passing 'input' and 'chat_history'. Pass an empty value for context so it is still defined in the prompt template.
    response = conversational_rag_chain.invoke(
        {"input": prompt_text},
        config={"configurable": {"session_id": "1"}}
    )

    st.chat_message("assistant").write(response["answer"])
    streamlit_chat_history.add_ai_message(response["answer"])

