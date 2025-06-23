from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    if not text.strip():
        return None
    try:
        embedding = model.encode(text, convert_to_tensor=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
