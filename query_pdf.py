import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class PDFQuerySystem:
    """RAG system for querying PDF documents"""
    
    def __init__(self, persist_directory="vector_db", model_name="gpt-3.5-turbo"):
        """
        Initialize the query system by loading the vector database.
        This is fast because embeddings are already computed.
        """
        print("Loading vector database...")
        
        # Load embeddings model
        self.embeddings = OpenAIEmbeddings()
        
        # Load the persisted vector database
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Create retriever
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,  # 0 for deterministic answers
            max_tokens=500
        )
        
        # Create prompt template
        template = """Answer the question based only on the following context. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Create the RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("System ready! You can now ask questions.")
    
    def ask(self, question):
        """
        Ask a question and get an answer based on the PDF content.
        """
        # Get answer
        answer = self.rag_chain.invoke(question)
        
        # Get source documents
        source_docs = self.retriever.invoke(question)
        
        return answer, source_docs
    
    def ask_with_sources(self, question):
        """
        Ask a question and display answer with source information.
        """
        print(f"\nQuestion: {question}")
        print("-" * 80)
        
        answer, sources = self.ask(question)
        
        print(f"Answer: {answer}\n")
        
        if sources:
            print("Sources:")
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get('page', 'unknown')
                content_preview = doc.page_content[:150].replace('\n', ' ')
                print(f"  [{i}] Page {page}: {content_preview}...")
        
        print("-" * 80)
        
        return answer

def interactive_mode():
    """Run in interactive mode for continuous querying"""
    
    # Initialize the system
    try:
        rag_system = PDFQuerySystem()
    except Exception as e:
        print(f"Error loading vector database: {e}")
        print("Make sure you've run index_pdf.py first!")
        return
    
    print("\n" + "="*80)
    print("RAG PDF Query System - Interactive Mode")
    print("="*80)
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        user_question = input("You: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_question:
            continue
        
        try:
            rag_system.ask_with_sources(user_question)
        except Exception as e:
            print(f"Error: {e}")

def single_query_mode(question):
    """Run a single query"""
    
    try:
        rag_system = PDFQuerySystem()
        answer = rag_system.ask_with_sources(question)
        return answer
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Run in interactive mode
    interactive_mode()
