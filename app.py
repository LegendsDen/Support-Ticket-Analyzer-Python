from flask import Flask, request, jsonify
from masking import full_masking_pipeline
from embedding import get_embedding

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Flask masking/embedding service!"

@app.route('/mask', methods=['POST'])
def mask_messages():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        masked_messages = []
        for msg in messages:
            masked = full_masking_pipeline(msg)
            if masked.strip():
                masked_messages.append(masked)
        return jsonify({"masked_messages": masked_messages})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/embed', methods=['POST'])
def embed_message():
    try:
        print("Received request for embedding")
        data = request.get_json()
        message = data.get('message', '')
        embedding = get_embedding(message)
        print("Generated embedding")
        # print(embedding.shape)
        return jsonify({"embedding": embedding})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
