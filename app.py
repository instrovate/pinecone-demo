import streamlit as st
import os
import pinecone_client as pinecone
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index = os.getenv("PINECONE_INDEX")

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index)

# UI
st.title("🔍 RAG App with Pinecone Vector DB")
query = st.text_input("Ask a question:")

# Sample content (you can load a text file or PDF here)
sample_data = {
    "1": "Microsoft Fabric is an all-in-one analytics solution for enterprises.",
    "2": "Power BI is a visualization tool within Microsoft Fabric.",
    "3": "Pinecone is a vector database used for semantic search."
}

# Step 1: Embed & upsert
if st.button("Embed & Upload Data"):
    for uid, text in sample_data.items():
        response = openai.Embedding.create(input=text, model="text-embedding-3-small")
        vector = response['data'][0]['embedding']
        index.upsert([(uid, vector, {"text": text})])
    st.success("Data embedded and stored in Pinecone.")

# Step 2: Query
if query:
    query_embed = openai.Embedding.create(input=query, model="text-embedding-3-small")
    query_vector = query_embed['data'][0]['embedding']
    result = index.query(vector=query_vector, top_k=2, include_metadata=True)

    st.subheader("🔎 Top Results")
    for match in result['matches']:
        st.write(f"Score: {match['score']}")
        st.write(f"Text: {match['metadata']['text']}")
