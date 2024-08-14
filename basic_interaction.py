# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:21:11 2024

@author: Admin
"""
import streamlit as st
from langchain.chat_models import ChatOpenAI



if 'conversation' not in st.session_state:
    st.session_state.conversation=[]



llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
)

#chat input always at the bottom of the screen
prompt = st.chat_input('Message to ChatBot...')


#if a prompt has been entered in the chat input
if prompt:
    st.session_state.conversation.append({'role':'user', 'content':prompt}) #add the user prompt to the conversation
    response=llm.invoke(prompt).content
    st.session_state.conversation.append({'role':'assistant', 'content': response}) #add the answer to the conversation

#display the conversation
for elem in st.session_state.conversation:
    with st.chat_message(elem['role']):
        st.write(elem['content'])