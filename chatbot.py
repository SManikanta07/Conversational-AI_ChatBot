import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["NVIDIA_API_KEY"] = "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"  

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

st.set_page_config(page_title="Simple Chatbot", layout="wide")
st.title("Let's Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])

chain = prompt_template | llm | StrOutputParser()

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        augmented_user_input = f"Question: {user_input}\n"

        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌") 
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
