import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for API tokens
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Create HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Streamlit app
st.title("RAG with Multi-PDF Uploader and Chat History")
st.write("Upload your PDF")

# Get the Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Create LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Chat Interface: Get session ID input
session_id = st.text_input("Session ID", value="default_session")

# Initialize chat history if not already in session state
if 'store' not in st.session_state:
    st.session_state.store = {}

# Upload PDF file
uploaded_file = st.file_uploader("Select a PDF file", type="pdf", accept_multiple_files=False)

if uploaded_file:
    # Process uploaded PDF file
    temppdf = "./temp.pdf"
    with open(temppdf, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name

    # Load documents from the PDF
    loader = PyPDFLoader(temppdf)
    documents = loader.load()

    # Split documents and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define prompts for context-aware retrieval and question-answering
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        " which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Question-answering system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Manage session-based chat history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Define the conversational chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Get user input for a question
    user_input = st.text_input("Your question:")

    if user_input:
        # Retrieve chat history and process user input through the conversational RAG chain
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        # Display the assistant's response and chat history
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)

else:
    st.warning("Please upload a PDF file")
