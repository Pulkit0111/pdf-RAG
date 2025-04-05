import streamlit as st # for implementing the UI
import os # for file operations
import tempfile # for creating a temporary directory
from langchain_community.document_loaders import PyMuPDFLoader # for loading the PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter # for splitting the documents
from langchain_openai import OpenAIEmbeddings # for embedding the documents
from langchain_community.vectorstores import FAISS # for creating a vector store
from langchain_openai import ChatOpenAI # for creating a chat model
from langchain import hub # for importing the prompt
from langchain_core.output_parsers import StrOutputParser # for parsing the output
from langchain_core.runnables import RunnablePassthrough # for running the chain
from dotenv import load_dotenv # for loading the environment variables
import time # for adding a delay

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="SmartPDF QA",
    page_icon="📚",
    layout="wide"
)

# App title and description
st.title("📚 SmartPDF QA")
st.markdown("Upload any PDF and ask questions about its content!")

# Initialize session state variables if they don't exist
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.TemporaryDirectory()
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to process the uploaded PDF
def process_pdf(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join(st.session_state.temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Show a spinner while processing
    with st.spinner(f"Processing {uploaded_file.name}... This may take a minute."):
        # Load the PDF
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # Initialize the embedding model
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Create a FAISS index
        vector_store = FAISS.from_documents(chunks, embedding_model)
        
        # Save to session state
        st.session_state.vector_store = vector_store
        st.session_state.pdf_processed = True
        st.session_state.file_name = uploaded_file.name
        
        return f"✅ {uploaded_file.name} processed successfully! ({len(chunks)} chunks created)"

# Create two columns for the layout
# col1, col2 = st.columns([1, 1])

# PDF Upload section
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
if uploaded_file is not None and (not st.session_state.pdf_processed or uploaded_file.name != st.session_state.file_name):
    result = process_pdf(uploaded_file)
    st.success(result)

# Question/Answer section
st.header("Ask Questions")    
# If a PDF has been processed, enable the question input
if st.session_state.pdf_processed and st.session_state.vector_store is not None:
    # Initialize the chat model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
    # Get the prompt template
    prompt = hub.pull("rlm/rag-prompt")
        
    # Initialize retriever
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        
    # Format docs function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Create the chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
        
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
        
    # Get user question
    user_question = st.chat_input("Ask a question about your PDF")
        
    if user_question:
        # Add user question to chat
        with st.chat_message("user"):
            st.write(user_question)
            
        # Generate answer with spinner
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Let me think..."):
                response = chain.invoke(user_question)
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        # Add interaction to chat history
        st.session_state.chat_history.append((user_question, response))          
else:
    st.info("Please upload a PDF document first.")

# Add a button to clear chat history
if st.session_state.chat_history:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    