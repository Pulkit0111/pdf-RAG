import streamlit as st # for implementing the UI
import os # for file operations
import tempfile # for creating a temporary directory
import time # for adding a delay
from langchain_community.document_loaders import PyMuPDFLoader # for loading the PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter # for splitting the documents
from langchain_openai import OpenAIEmbeddings # for embedding the documents
from langchain_community.vectorstores import FAISS # for creating a vector store
from langchain_openai import ChatOpenAI # for creating a chat model
from langchain import hub # for importing the prompt
from langchain_core.output_parsers import StrOutputParser # for parsing the output
from langchain_core.runnables import RunnablePassthrough # for running the chain
from dotenv import load_dotenv # for loading the environment variables
from langchain_tavily import TavilySearch # for searching the web
from langchain_core.prompts import ChatPromptTemplate # for creating custom prompts

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="SmartPDF QA",
    page_icon="📚",
    layout="wide"
)

# Initialize the TavilySearch tool
tool = TavilySearch(
    max_results=3,
    topic="general"
)

# Initialize the chat model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# App title and description
st.title("📚 SmartPDF QA")
st.markdown("Upload any PDF and ask questions about its content! If the answer isn't in the PDF, the app will search the web for you.")

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
        chunks = text_splitter.split_documents(docs)
        
        # Create a FAISS index
        vector_store = FAISS.from_documents(chunks, embedding_model)
        
        # Save to session state
        st.session_state.vector_store = vector_store
        st.session_state.pdf_processed = True
        st.session_state.file_name = uploaded_file.name
        
        return f"✅ {uploaded_file.name} processed successfully! ({len(chunks)} chunks created)"

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
    # Get the prompt template
    prompt = hub.pull("rlm/rag-prompt")
        
    # Initialize retriever
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        
    # Format docs function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create custom prompt for determining if PDF content has the answer
    answer_determination_prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant tasked with determining if the provided context from a PDF contains sufficient information to answer a user's question.

    Context from PDF: {context}
    
    User Question: {question}
    
    First, carefully analyze if the context provides adequate information to answer the question.
    
    If the context contains sufficient information to answer the question, respond with a complete and accurate answer based ONLY on the provided context.
    
    If the context does NOT contain sufficient information to fully answer the question, respond with exactly: "[NEED_WEB_SEARCH]"
    
    Your response:
    """)
    
    # Create the initial RAG chain that determines if PDF content is sufficient
    determination_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | answer_determination_prompt
        | llm
        | StrOutputParser()
    )
    
    # Create a web search chain
    web_search_prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant helping a user with their question.
    
    User Question: {question}
    
    Web Search Results: {web_results}
    
    Using the web search results, provide a comprehensive and accurate answer to the user's question.
    Make sure to cite sources from the search results where appropriate.
    """)
    
    # Define the web search chain
    web_search_chain = (
        {
            "question": RunnablePassthrough(),
            "web_results": lambda x: tool.invoke({"query": x})
        }
        | web_search_prompt
        | llm
        | StrOutputParser()
    )
    
    # Define the combined chain that determines whether to use PDF content or web search
    def agentic_chain(question):
        # First try to answer from the PDF
        pdf_response = determination_chain.invoke(question)
        
        # Check if web search is needed
        if "[NEED_WEB_SEARCH]" in pdf_response:
            st.info("Information not found in PDF. Searching the web...")
            return web_search_chain.invoke(question)
        else:
            return pdf_response
        
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
                response = agentic_chain(user_question)
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
    