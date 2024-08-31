from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain


# Set up the LLM
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
)

conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_message_template = '''
DEFINE HERE PROMPT 
'''

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

def make_test(testRequisite):
    print(testRequisite)
#tool template that was used for making test TO CHANGE 
tools = [
    Tool(
        name="  ",
        func=make_test,
        description="Useful when you need to make a test for students. The input will be the test requisite that are: number of questions, difficulty."
    )
]

# Initialize agent with tools and the custom prompt
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=conversational_memory,
    handle_parsing_error=True,
    agent_kwargs={
        "system_message": system_message_template,
        "memory_prompts": [MessagesPlaceholder(variable_name="chat_history")],
        "input_variables": ["input", "agent_scratchpad", "chat_history"]
    }
)

st.title("CHATBOT MODEL") 

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load previous messages into memory
for message in st.session_state.messages:
    if message["role"] == "user":
        conversational_memory.chat_memory.add_user_message(message["content"])
    elif message["role"] == "assistant":
        conversational_memory.chat_memory.add_ai_message(message["content"])

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_input := st.chat_input("What kind of test questions do you need?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = agent.run(input=user_input)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Update conversation memory
    conversational_memory.chat_memory.add_user_message(user_input)
    conversational_memory.chat_memory.add_ai_message(full_response)
    
    
  