import re
import streamlit as st
import pdfplumber
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from concurrent.futures import ThreadPoolExecutor

# LLM CONFIGURATION
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
)

# MEMORY CONFIGURATION
conversational_memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, input_key="input"
)


# PDF FUNCTION
def extract_text_from_pdf(file):
    texts_with_pages = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                texts_with_pages.append((text, i + 1))  # Store text with page number
    return texts_with_pages


def process_pdfs(files):
    all_texts_with_pages = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_text_from_pdf, file) for file in files]
        for future in futures:
            all_texts_with_pages.extend(future.result())
    return all_texts_with_pages


# RAG FUNCTION
def build_rag_chain(all_texts_with_pages):
    texts_with_metadata = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

    for text, page_num in all_texts_with_pages:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            texts_with_metadata.append({"text": chunk, "metadata": {"page": page_num}})

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_texts(
        [text["text"] for text in texts_with_metadata],
        embedding_model,
        metadatas=[text["metadata"] for text in texts_with_metadata],  # Store metadata
    )
    retriever = vectorstore.as_retriever()

    # PROMPT
    system_prompt = """
        You are a specialized assistant designed to generate test questions for student exams or tests.
        I want under the question, the answer and the number corespondent to the question (ex: "Answer 1"), and for multiple-choice questions, provide options labeled A, B, C, and D.
        Before each question you must give the number of the questions (ex: "Question 1")
        If it is only one question you also must have the number of the question (ex: "Question 1") 
        When you generate the answer, write down the page number where the answer is in the pdf file.
        \n\n
        {context}
        {chat_history}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llmchain = LLMChain(llm=llm, prompt=prompt, memory=conversational_memory)
    rag_chain = create_retrieval_chain(retriever, llmchain)
    return rag_chain


# Function to extract generated questions and options
def extract_generated_questions_options():
    questions = ""
    for message in st.session_state.chats[st.session_state.active_chat]:
        if message["role"] == "assistant":
            question_blocks = re.findall(
                r"(Question \d+.*?\n(?:[A-D]\) .*?\n)+)", message["content"], re.DOTALL
            )
            for block in question_blocks:
                questions += f"{block.strip()}\n\n"
    return questions


# Function to extract generated answers
def extract_generated_answers():
    answers = ""
    for message in st.session_state.chats[st.session_state.active_chat]:
        if message["role"] == "assistant":
            answer_blocks = re.findall(
                r"Answer \d+.*?(?=\n|$)", message["content"], re.DOTALL
            )
            for block in answer_blocks:
                answers += f"{block.strip()}\n"
    return answers


# Streamlit CONFIGURATION
st.set_page_config(layout="wide")

# INITIALISING SESSION STATES
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Chat 1"
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = []

# Sidebar
with st.sidebar:
    st.title("Test Question Generator")
    with st.expander("Menu"):
        if st.button("New chat"):
            new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
            st.session_state.chats[new_chat_name] = []
            st.session_state.active_chat = new_chat_name

        if st.button("Delete chat"):
            if len(st.session_state.chats) > 1:
                del st.session_state.chats[st.session_state.active_chat]
                st.session_state.active_chat = list(st.session_state.chats.keys())[0]
            else:
                st.warning("Cannot delete the last remaining chat.")

    with st.expander("Chats"):
        for chat_name in st.session_state.chats.keys():
            if st.button(chat_name):
                st.session_state.active_chat = chat_name

    with st.expander("Download options"):
        st.download_button(
            label="Download Generated Questions and Options",
            data=extract_generated_questions_options(),
            file_name=f"{st.session_state.active_chat}_questions_with_options.txt",
            mime="text/plain",
        )
        st.download_button(
            label="Download Generated Answers",
            data=extract_generated_answers(),
            file_name=f"{st.session_state.active_chat}_answers.txt",
            mime="text/plain",
        )


# MAIN FUNCTION
def main():
    # Initialize session state variables if not already initialized
    if "chats" not in st.session_state:
        st.session_state.chats = {"default": []}  # Initializes chat history
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = "default"  # Sets default active chat
    if "pdf_texts_with_pages" not in st.session_state:
        st.session_state.pdf_texts_with_pages = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # PDF UPLOADING
    with st.form("test_form"):
        uploaded_files = st.file_uploader(
            "Upload PDF files", type="pdf", accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.pdf_texts_with_pages = process_pdfs(uploaded_files)
            st.success("PDF content loaded successfully!")

        # FORM GENERATION
        with st.expander("Configure your test questions!"):
            num_questions = st.number_input(
                "Enter the number of questions:", min_value=1, step=1
            )
            difficulty = st.selectbox(
                "Select the difficulty level:", ["Easy", "Medium", "Hard"]
            )
            submitted = st.form_submit_button("Generate Test")

    if submitted:
        if not st.session_state.pdf_texts_with_pages:
            st.warning("Please upload a PDF file before generating the test.")
        else:
            user_input = f"Generate {num_questions} {difficulty} questions based on the uploaded PDFs."
            st.session_state.chats[st.session_state.active_chat].append(
                {"role": "user", "content": user_input}
            )
            st.session_state.rag_chain = build_rag_chain(
                st.session_state.pdf_texts_with_pages
            )
            response = st.session_state.rag_chain.invoke({"input": user_input})[
                "answer"
            ]["text"]
            st.session_state.chats[st.session_state.active_chat].append(
                {"role": "assistant", "content": response}
            )

    # CHAT HISTORY
    for elem in st.session_state.chats[st.session_state.active_chat]:
        with st.chat_message(elem["role"]):
            st.write(elem["content"])

    # Chat input at the bottom of the screen
    user_input = st.chat_input("Insert text here.", key="main_chat_input")

    if user_input:
        st.session_state.chats[st.session_state.active_chat].append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = st.session_state.rag_chain.invoke({"input": user_input})[
                "answer"
            ]["text"]
            message_placeholder.markdown(full_response)

        st.session_state.chats[st.session_state.active_chat].append(
            {"role": "assistant", "content": full_response}
        )

        # Update conversation memory
        conversational_memory.chat_memory.add_user_message(user_input)
        conversational_memory.chat_memory.add_ai_message(full_response)


if __name__ == "__main__":
    main()
