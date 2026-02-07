from sentence_transformers import SentenceTransformer

# uses all-MiniLM-L6-v2 to return embeddings

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(sentences):
    return model.encode(sentences)