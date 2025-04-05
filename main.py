from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

# Path of the PDF file
file_path = "./data/numpy.pdf"

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
prompt = hub.pull("rlm/rag-prompt")

# Initialize the chat model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Check if FAISS index exists
if os.path.exists("faiss_index"):
    # Load existing FAISS index
    print("Loading existing FAISS index...")
    db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    # Load and process the PDF if index doesn't exist
    print("Creating new FAISS index...")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Splitting {len(docs)} documents into {len(chunks)} chunks...")
    
    # Create a FAISS index
    db = FAISS.from_documents(chunks, embedding_model)
    
    # Save the FAISS index to a local file
    db.save_local("faiss_index")
    print("FAISS index saved to ./faiss_index/")

# Initialize retriever with top 3 results
retriever = db.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Run the PDF bot
while True:
    question = input("Enter a question: ")
    if question.lower() == "exit":
        print("GoodBye!")
        break
    response = chain.invoke(question)
    print(f"\nPDF answer: {response}\n")

