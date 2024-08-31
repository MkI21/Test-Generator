import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import UnstructuredPDFLoader

llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
)

conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_message_template = '''
You are a specialized assistant designed to generate test questions for student exams or tests. Your primary task is to create relevant, clear, and appropriate questions based on the information provided by the user. Here are your guidelines:

1. Always focus on generating test questions. This is your main and only purpose.
2. Ask for clarification if the user doesn't provide enough information about the subject, difficulty level, or question type.
3. Generate questions that are appropriate for the specified subject and difficulty level.
4. If asked, provide a mix of question types (e.g., multiple choice, short answer, essay) unless otherwise specified.
5. Ensure that the questions are clear, unambiguous, and relevant to the subject matter.
6. If requested, provide answer keys or explanations for the questions.
7. Do not engage in conversations or tasks unrelated to generating test questions.

Remember, your responses should always be in the context of creating test questions. If a user asks for anything else, politely redirect them to your primary function of generating test questions.
'''

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

def make_test(testRequisite):
    return f"Generating a test with the following requirements: {testRequisite}"

tools = [
    Tool(
        name="create test",
        func=make_test,
        description="Use this to make a test for students. The input should include the test requisites such as number of questions, difficulty level, and topic."
    )
]

# Initialize agent with tools and the custom prompt
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=conversational_memory,
    handle_parsing_error=True,
    agent_kwargs={"system_message": system_message_template}
)

st.title("Test Question Generator")



if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user inputs for test generation
with st.form("test_form"):
    topic = st.text_input("Enter the topic for the test questions:")
    num_questions = st.number_input("Enter the number of questions:", min_value=1, step=1)
    difficulty = st.selectbox("Select the difficulty level:", ["Easy", "Medium", "Hard"])
    question_type = st.selectbox("Select the type of questions:", ["Multiple Choice", "Short Answer", "Essay", "Mix"])
    submitted = st.form_submit_button("Generate Test")

if submitted:
    user_input = f"Generate {num_questions} {difficulty} {question_type} questions on the topic of {topic}."
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = agent.run(input=user_input)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
