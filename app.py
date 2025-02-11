

from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.planners import create_history_aware_retriever
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.memory import BaseChatMessageHistory, ChatMessageHistory

import streamlit as st
from langchain.vectorstores import Chroma

# 1.  Model Setup

st.session_state.model = ChatHuggingFace(
    api_key=st.secrets.huggingfacetoken, model="microsoft/Phi-3-mini-4k-instruct"
)


# 4. Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are WiChat, the chatbot for the Worldbank Ideas Project. You are friendly and follow instructions to answer questions extremely well. Please be truthful and give direct answers. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the response short and concise in at most five sentences. If the user chats in a different language, translate accurately and respond in the same language. You will provide specific details and accurate answers to user queries on the Worldbank Ideas Project."),
         MessagesPlaceholder("chat_history"),
        ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question.")
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

def create_history_aware_retriever(llm, retriever, contextualize_q_prompt):
    """Create a chain to generate the standalone question for history aware retriever."""
    from langchain_core.runnables import chain

    template = contextualize_q_prompt

    def generate_standalone_question(input, chat_history):
        messages = template.format_messages(
            input=input,
            chat_history=chat_history
        )

        for m in messages:
            if m.type == "human":
                messages[messages.index(m)] = HumanMessage(content = m.content)
            if m.type == "ai":
                messages[messages.index(m)] = AIMessage(content = m.content)
            
            standalone_question = llm.invoke(messages).content
            return standalone_question
        
        retriever_chain = RunnablePassthrough.assign(
            standalone_question=generate_standalone_question
        ) | retriever

        return retriever_chain

# 7. RAG Chain construction
@st.cache_resource
def store_data(_data, _embeddings):
  # vector store
  db = FAISS.from_documents(data, embeddings)
  return db

embeddings = embed(data, 'cpu', 'sentence-transformers/all-MiniLM-L6-v2')
db = store_data(data, embeddings)


retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
history_aware_retriever = create_history_aware_retriever(st.session_state.model, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(st.session_state.model, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.runnables import chain
from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": history_aware_retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | st.session_state.model
)

conversational_rag_chain = RunnablePassthrough.assign(
    answer = rag_chain
)

# 8. Streamlit UI
st.title("WiChat")
st.markdown("Welcome!")

# 9. Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"  # Or generate a unique ID

# 10. Chat Input
if prompt := st.chat_input(placeholder="Ask me anything!"):
    st.session_state.prompt = prompt  #Save input text
    st.chat_message("user").write(prompt)
    st.session_state.past_messages = get_session_history(st.session_state.session_id).messages # load messages with custom function

    # Format chat history correctly
    chat_history = get_session_history(st.session_state.session_id).messages
    formatted_chat_history = []
    for msg in chat_history:
        if msg.type == "human":
            formatted_chat_history.append(HumanMessage(content = msg.content))
        if msg.type == "ai":
            formatted_chat_history.append(AIMessage(content = msg.content))

    with st.spinner():
        response = conversational_rag_chain.invoke(
            {"input": st.session_state.prompt, "chat_history": formatted_chat_history},
            config={"configurable": {"session_id": "1"}},
        )

    #11 Get assistant history and print
    st.chat_message("assistant").write(response["answer"].content)
    streamlit_chat_history = get_session_history(st.session_state.session_id)

    #12. Print all messages at the end:
    if "session_id" in st.session_state:
        messages = get_session_history(st.session_state.session_id).messages
        for msg in messages:
            st.chat_message(msg.type).write(msg.content)
