import streamlit as st
# No need for 'os' if all secrets are handled by st.secrets
# No need for 'dotenv' if all secrets are handled by st.secrets
from pinecone import Pinecone, Index
import openai

# Load environment variables from Streamlit secrets
# Streamlit automatically makes secrets available via st.secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENV"] # Still good to retrieve if used for clarity or other purposes
pinecone_index = st.secrets["PINECONE_INDEX"]

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Connect to the index
index = pc.Index(pinecone_index)

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
