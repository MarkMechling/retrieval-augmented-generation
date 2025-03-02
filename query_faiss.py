import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Load the FAISS index and text chunks
print("Loading FAISS index and metadata...")
index = faiss.read_index("gdpr_vector_db.index")
with open("gdpr_vector_db_texts.npy", "rb") as f:
    gdpr_chunks = np.load(f, allow_pickle=True)
print("FAISS index and metadata loaded successfully!")

# Step 2: Load the SentenceTransformer model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure the same model used for embeddings
print("Embedding model loaded successfully!")


# Step 3: Define a query function
def query_faiss(query, top_k=5):
    # Generate embeddings for the query
    query_embedding = embedding_model.encode([query])

    # Perform similarity search on the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the top matching chunks
    results = [gdpr_chunks[i] for i in indices[0]]
    return results


# Step 4: Test the query function
if __name__ == "__main__":
    query_text = "Under what circumstances is the transfer of data into another country not allowed?"
    print(f"Query: {query_text}\n")

    # Retrieve results
    matches = query_faiss(query_text, top_k=3)
    print("Top Matches:")
    for i, match in enumerate(matches, start=1):
        print(f"\nMatch {i}:\n{match}")
