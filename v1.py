import streamlit as st
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.memory import ConversationBufferMemory  # type: ignore

# Initialize session states for chats and the active chat index
if 'chats' not in st.session_state:
    st.session_state.chats = {'Chat 1': []}
if 'active_chat' not in st.session_state:
    st.session_state.active_chat = 'Chat 1'

# Configurarea LLM
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
)

conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to compile chat history into a downloadable text
def download_chat():
    chat_history = ""
    for message in st.session_state.chats[st.session_state.active_chat]:
        role = "User" if message['role'] == "user" else "Assistant"
        chat_history += f"{role}: {message['content']}\n"
    return chat_history

# Funcție pentru a procesa fișierul PDF și a extrage textul
def process_pdf(file):
    pdf_reader = PdfReader(file)
    pdf_text = ''
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

# Funcție pentru a construi lanțul RAG
def build_rag_chain(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(pdf_text)

    # Crearea vectorstore folosind Chroma și HuggingFaceEmbeddings
    vectorstore = Chroma.from_texts(splits, HuggingFaceEmbeddings())

    # Crearea retriever-ului
    retriever = vectorstore.as_retriever()

    # Definirea promptului de sistem
    system_prompt = '''
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, say that you don't know.
        \n\n
        {context}
    '''

    # Configurarea promptului
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Construirea lanțului de întrebări-răspunsuri și lanțului RAG
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

st.set_page_config(layout="wide")
st.title("Test Question Generator")

# Sidebar for selecting and managing chats
with st.sidebar:
    if st.button("New chat"):
        new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.active_chat = new_chat_name
    
    with st.expander("Chats"):
        for chat_name in st.session_state.chats.keys():
            if st.button(chat_name):
                st.session_state.active_chat = chat_name

    # Button to delete the active chat
    if st.button("Delete chat"):
        if len(st.session_state.chats) > 1:  # Ensure at least one chat remains
            del st.session_state.chats[st.session_state.active_chat]
            st.session_state.active_chat = list(st.session_state.chats.keys())[0]  # Switch to the first available chat
        else:
            st.warning("Cannot delete the last remaining chat.")


# Funcția principală a aplicației Streamlit
def main():
    global pdf_text
    with st.expander("Configure your test questions!"):
        # File uploader in the form
        with st.form("test_form"):
            uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
            if uploaded_file is not None:
                pdf_text = process_pdf(uploaded_file)
                st.success("PDF content loaded successfully!")

            # Capture user inputs for test generation
            num_questions = st.number_input(
                "Enter the number of questions:", min_value=1, step=1
            )
            difficulty = st.selectbox(
                "Select the difficulty level:", ["Easy", "Medium", "Hard"]
            )
            question_type = st.selectbox(
                "Select the type of questions:",
                ["Multiple Choice", "Short Answer", "Essay", "Mix"],
            )
            submitted = st.form_submit_button("Generate Test")

    if submitted and uploaded_file:
        user_input = f"Generate {num_questions} {difficulty} {question_type} questions based on the uploaded PDF."
        st.session_state.chats[st.session_state.active_chat].append({"role": "user", "content": user_input})
        
        rag_chain = build_rag_chain(pdf_text)
        
        # Generarea întrebărilor folosind lanțul RAG
        response = rag_chain.invoke({"input": user_input})
        
        # Extrage răspunsul ca șir de caractere
        response_text = response.get("answer", "Sorry, I couldn't generate a response.")
        
        # Afișează răspunsul în interfață
        st.session_state.chats[st.session_state.active_chat].append({"role": "assistant", "content": response_text})

        # Actualizează conversația în interfață
        for elem in st.session_state.chats[st.session_state.active_chat]:
            with st.chat_message(elem['role']):
                st.write(elem['content'])

    # Add a download button for the chat history
    chat_history = download_chat()
    st.download_button(
        label="Download Chat",
        data=chat_history,
        file_name=f"{st.session_state.active_chat}_history.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
