from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load the GDPR text
with open("gdpr_full_text.txt", "r", encoding="utf-8") as file:
    gdpr_text = file.read()

# Step 2: Split the text into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,  # Smaller chunk size for better granularity
    chunk_overlap=50,  # Overlap to maintain context between chunks
)
gdpr_chunks = text_splitter.split_text(gdpr_text)

print(f"Number of chunks created: {len(gdpr_chunks)}")

# Step 3: Load a local embedding model (SentenceTransformers)
print("Loading local embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models too
print("Local embedding model loaded successfully!")

# Step 4: Generate embeddings for the chunks
print("Generating embeddings for text chunks...")
gdpr_embeddings = embedding_model.encode(gdpr_chunks, show_progress_bar=True)
print("Embeddings generated successfully!")

# Step 5: Create a FAISS index
print("Creating FAISS vector store...")
dimension = gdpr_embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(gdpr_embeddings)  # Add embeddings to the index
print("FAISS index created successfully!")

# Step 6: Save the FAISS index and metadata
faiss.write_index(index, "gdpr_vector_db.index")
with open("gdpr_vector_db_texts.npy", "wb") as f:
    np.save(f, gdpr_chunks)

print("FAISS vector store and metadata saved locally!")
