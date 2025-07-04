# from sentence_transformers import SentenceTransformer
# import torch

# model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_embedding(text):
#     if not text.strip():
#         return None
#     try:
#         embedding = model.encode(text, convert_to_tensor=True)
#         return embedding.tolist()
#     except Exception as e:
#         print(f"Error generating embedding: {e}")
#         return None



from sentence_transformers import SentenceTransformer
import torch

# Load the mpnet model
model = SentenceTransformer('all-mpnet-base-v2')

def get_embedding(text):
    if not text.strip():
        return None
    try:
        # Generate embedding as a tensor and convert to list
        embedding = model.encode(text, convert_to_tensor=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample sentence for embedding."
    vector = get_embedding(sample_text)
    print(vector)
