import streamlit as st
import openai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI  # Correct import for OpenAI models

# 1. API Key and Model Setup
openai.api_key = st.secrets["OPENAI_API_KEY"]
if not openai.api_key:
    st.warning("OpenAI API key not found. Please add it to streamlit secrets.")
    st.stop()

MODEL = "gpt-4o"  # Or whatever model you want to use

# 2. Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are WiChat, the chatbot for the Worldbank Ideas Project. You are friendly and follow instructions to answer questions extremely well. Be truthful and give direct answers. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the response short and concise in at most five sentences. If the user chats in a different language, translate accurately and respond in the same language. You will provide specific details and accurate answers to user queries on the Worldbank Ideas Project."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}") #Removed Context part here
    ]
)

# 3. Create Chat Model
llm = ChatOpenAI(model_name=MODEL, openai_api_key=openai.api_key, temperature=0.7)

# 4. RAG chain Function (Leveraging LangChain)
def retrieval_augmented_generation(chain, prompt_input, chat_history, context=""):
    messages = StreamlitChatMessageHistory(key="langchain_messages").messages
    try:
        response = chain.invoke({"input": prompt_input, "context": context}, config={"run_name": "Assistant", "configurable": {"session_id": 12345}})
        return response.content
    except Exception as e:
        return f"Error generating response: {e}"

# 5. Streamlit Chat Interface
st.title("WiChat")
st.markdown("Welcome!!! WiChat wants to know what's on your mind")

# Initialize Streamlit chat history (This must be outside any function)
chat_history = StreamlitChatMessageHistory(key="langchain_messages")
if len(chat_history.messages) == 0:
    chat_history.add_ai_message("Hello! How can I help you today?") #Add initial prompt to start

# Chat input
if prompt_text := st.chat_input():
    st.chat_message("user").write(prompt_text)

# 6 Create RAG Chain
chat_chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(chat_chain, lambda session_id: StreamlitChatMessageHistory(key="langchain_messages"), input_messages_key="input",history_messages_key="chat_history")
st.session_state.past_messages = StreamlitChatMessageHistory(key="langchain_messages").messages

# Generate and display assistant response
if prompt_text:
    with st.chat_message("assistant"):
        with st.spinner():
            st.session_state.assistant_response = chain_with_history.invoke({"input": prompt_text})
            st.write(st.session_state.assistant_response["content"])
        #After response add messgae for next itteration:
        chain_with_history.get_messages_from_history()
        StreamlitChatMessageHistory().add_ai_message(prompt_text)

# Display chat messages from history (Put After Response so it occurs once)
history = StreamlitChatMessageHistory(key="langchain_messages")
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)
