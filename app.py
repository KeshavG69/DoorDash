__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



import streamlit as st
from main import *


st.set_page_config(page_title="Doordash Dasher Support 🍽️📦")

if "messages" not in st.session_state.keys():
    
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    

    

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


    
if prompt := st.chat_input("Send a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            response = app.invoke ({'query':prompt,'chat_history':st.session_state.chat_history})
            st.markdown(response['generation']) 
    message = {"role": "assistant", "content": response['generation']}
    st.session_state.messages.append(message)
