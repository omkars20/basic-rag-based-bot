import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def index_pdf(pdf_path, persist_directory="vector_db"):
    """
    Load a PDF, split it into chunks, create embeddings, and store in vector DB.
    This should be run once when you have a new PDF or want to update the index.
    """
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Loading PDF: {pdf_path}")
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages from PDF")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    
    # Create embeddings
    print("Creating embeddings... (this may take a while)")
    embeddings = OpenAIEmbeddings()
    
    # Create and persist vector store
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector database created and saved to: {persist_directory}")
    print("Indexing complete!")
    
    return vectordb

if __name__ == "__main__":
    # Your PDF filename
    pdf_file = "https__www.maximusveritas.com_wp-content_uploads_2017_09_Marcus-Aurelius-Meditations.pdf"
    
    try:
        index_pdf(pdf_file)
    except Exception as e:
        print(f"Error: {e}")
