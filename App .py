pip install sqlite3 pandas scikit-learn sentence-transformers chromadb scipy

import sqlite3
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

# Initialize SentenceTransformer model for semantic search
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  

# Initialize ChromaDB client
client = chromadb.Client(Settings())

# Function to clean subtitle text
def clean_subtitle(text):
    # Remove timestamps and other unnecessary characters
    text = re.sub(r'\d{2}:\d{2}:\d{2}.*-->', '', text)  # Remove timestamps
    text = re.sub(r'\[.*?\]', '', text)  # Remove metadata like [Music], [Laughing]
    text = re.sub(r'\n+', ' ', text)  # Replace newline characters with space
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()

# Step 1: Ingest Data from SQLite Database and Clean it
def ingest_subtitle_data(db_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    
    # Query the database to get subtitle data
    query = "SELECT * FROM subtitles"  # Modify this query based on the actual table/column names
    df = pd.read_sql_query(query, conn)
    
    # Clean the subtitle text
    df['clean_subtitle'] = df['subtitle'].apply(clean_subtitle)
    
    # Close the database connection
    conn.close()
    
    return df

# Step 2: Keyword-based (TF-IDF) Vectorization
def keyword_based_vectorizer(cleaned_documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cleaned_documents)  # TF-IDF matrix
    return vectorizer, tfidf_matrix

# Step 3: Semantic-based Embeddings using SentenceTransformers
def semantic_vectorizer(cleaned_documents):
    document_embeddings = semantic_model.encode(cleaned_documents, convert_to_tensor=True)
    return document_embeddings

# Step 4: Chunking documents to avoid information loss (for large documents)
def chunk_document(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Step 5: Store embeddings in ChromaDB
def store_in_chromadb(cleaned_documents, embeddings):
    collection = client.create_collection("subtitle_embeddings")
    collection.add_embeddings(embeddings=embeddings, documents=cleaned_documents)
    return collection

# Step 6: Search for relevant subtitles using cosine similarity
def search_by_keyword(query, vectorizer, tfidf_matrix):
    query_vec = vectorizer.transform([query])  # Vectorize the query
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix)  # Calculate similarity
    return cosine_sim

def search_by_semantics(query, collection):
    query_embedding = semantic_model.encode([query], convert_to_tensor=True)
    results = collection.query(query_embedding, n_results=5)  # Query ChromaDB for top results
    return results

# Example usage:
# File ingestion from SQLite database
subtitle_data = ingest_subtitle_data('subtitles.db')  # Path to your .db file

# Keyword-Based Search
tfidf_vectorizer, tfidf_matrix = keyword_based_vectorizer(subtitle_data['clean_subtitle'])

# Semantic Search (using SentenceTransformers)
semantic_embeddings = semantic_vectorizer(subtitle_data['clean_subtitle'])

# Store in ChromaDB for future retrieval
collection = store_in_chromadb(subtitle_data['clean_subtitle'], semantic_embeddings)

# Test keyword-based search
keyword_query = "explosion in the scene"
keyword_results = search_by_keyword(keyword_query, tfidf_vectorizer, tfidf_matrix)

# Test semantic-based search
semantic_query = "a big blast in the movie"
semantic_results = search_by_semantics(semantic_query, collection)

# Display results
print("Keyword-Based Search Results:", keyword_results)
print("Semantic-Based Search Results:", semantic_results)
