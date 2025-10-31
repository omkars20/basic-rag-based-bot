from langchain_community.vectorstores import Chroma

# Path to your persisted database
persist_directory = "vector_db"

# Load the Chroma vector database
vectordb = Chroma(persist_directory=persist_directory)

# Get all documents stored in the database
# Note: You may need to set the embedding_function if you see errors when retrieving with the full setup.
all_docs = vectordb.get(include=["documents"])

for i, doc_text in enumerate(all_docs["documents"]):
    print(f"\n--- Document {i+1} ---\n{doc_text}\n")
