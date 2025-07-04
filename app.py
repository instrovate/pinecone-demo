import streamlit as st
import openai
import chromadb
from chromadb.utils import embedding_functions

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize ChromaDB client
# For an in-memory client (data is lost on app restart):
client = chromadb.Client()
# Or for a persistent client (data saved to a file in the app directory):
# client = chromadb.PersistentClient(path="/app/chroma_db") # or "./chroma_db"

# Define OpenAI embedding function for ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

# Get or create a collection
collection_name = "my_rag_collection"
try:
    collection = client.get_collection(name=collection_name, embedding_function=openai_ef)
except Exception: # If collection doesn't exist, create it
    collection = client.create_collection(name=collection_name, embedding_function=openai_ef)

# UI
st.title("🔍 RAG App with ChromaDB")
query = st.text_input("Ask a question:")

# Sample content (you can load a text file or PDF here)
sample_data = {
    "1": "Microsoft Fabric is an all-in-one analytics solution for enterprises.",
    "2": "Power BI is a visualization tool within Microsoft Fabric.",
    "3": "ChromaDB is a vector database used for semantic search."
}

# Step 1: Embed & upsert
if st.button("Embed & Upload Data"):
    # Prepare documents for ChromaDB
    documents = [text for text in sample_data.values()]
    metadatas = [{"text": text} for text in sample_data.values()]
    ids = [uid for uid in sample_data.keys()]

    # Add documents to ChromaDB (it will handle embedding using the function we set)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    st.success("Data embedded and stored in ChromaDB.")

# Step 2: Query
if query:
    # ChromaDB will embed the query using the same embedding function
    result = collection.query(
        query_texts=[query],
        n_results=2,
        include=['documents', 'metadatas', 'distances'] # 'distances' is similar to 'score'
    )

    st.subheader("🔎 Top Results")
    if result['documents'] and result['documents'][0]:
        for i in range(len(result['documents'][0])):
            st.write(f"Distance (lower is better): {result['distances'][0][i]}")
            st.write(f"Text: {result['documents'][0][i]}")
    else:
        st.write("No results found.")
