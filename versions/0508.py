from langchain.chat_models import ChatOpenAI # type: ignore
import streamlit as st # type: ignore
from langchain.agents import initialize_agent, AgentType # type: ignore
from langchain.agents import Tool # type: ignore
from langchain.chains.conversation.memory import ConversationBufferMemory # type: ignore
from langchain.prompts import MessagesPlaceholder # type: ignore
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain_community.document_loaders import UnstructuredPDFLoader
from PyPDF2 import PdfReader

# Set up the LLM
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
)

conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_message_template = '''
You are a specialized assistant designed to generate test questions for student exams or tests.
i want under the question , the answer 
'''

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

def process_pdf(file):
    # Load the PDF data
    pdf_reader = PdfReader(file)
    pdf_text = ''
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

pdf_text = ''

def make_test(testRequisite):
    global pdf_text
    print(testRequisite)
    print("PDF content:", pdf_text)
    if not pdf_text:
        return "Apologies, but I didn't receive any information to work with. Please provide the PDF content or a link to it, and I'll help you generate 2 easy questions."
    # Your logic to create questions based on pdf_text and testRequisite
    return f"Generating questions based on the PDF content: {pdf_text[:500]}..."

tools = [
    Tool(
        name="Generator_test",
        func=make_test,
        description="Useful when you need to make a test for students. The input will be the file that the user add, in that case a pdf ."
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

st.set_page_config(layout="wide")

# Main function
def main():
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    global pdf_text
    if uploaded_file is not None:
        st.write("You have uploaded a file!")
        pdf_text = process_pdf(uploaded_file)
        st.write("PDF content loaded successfully!")

    # Create a sidebar for chat selection
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        with st.expander("Menu"):
            if st.button("Saved"):
                st.session_state.page = "saved"
            if st.button("Favorites"):
                st.session_state.page = "favorites"
            if st.button("Downloads"):
                st.session_state.page = "downloads"
            if st.button("Archives"):
                st.session_state.page = "archives"

        if st.button("New chat"):
            st.session_state.page = "New chat"

    with col2:
        st.title("Questions")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Insert text here"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = agent.run(input=user_input)
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            conversational_memory.chat_memory.add_user_message(user_input)
            conversational_memory.chat_memory.add_ai_message(full_response)

if __name__ == '__main__':
    main()
