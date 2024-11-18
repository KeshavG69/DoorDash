import streamlit as st
from main import *


st.set_page_config(page_title="🤗💬 HugChat")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


    
if prompt := st.chat_input("Send a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_history=st.session_state.messages
            response = app.invoke ({'query':prompt,'chat_history':st.session_state.messages})
            st.markdown(response['generation']) 
    message = {"role": "assistant", "content": response['generation']}
    st.session_state.messages.append(message)