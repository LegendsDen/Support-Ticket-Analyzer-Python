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
        return jsonify({"embedding": embedding})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/embed_batch', methods=['POST'])
def embed_batch():
    try:
        data = request.get_json()
        batches = data.get("batches", [])  

        results = []
        for item in batches:
            ticketId = item.get("ticketId")
            message = item.get("message", "")
            embedding = get_embedding(message)
            results.append({
                "ticketId": ticketId,
                "embedding": embedding
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/mask_batch', methods=['POST'])
def mask_batch():
    try:
        data = request.get_json()
        batches = data.get('batches', [])  

        results = []
        for item in batches:
            ticketId = item.get("ticketId")
            messages = item.get("messages", [])
            masked = [full_masking_pipeline(m) for m in messages if m.strip()]
            results.append({"ticketId": ticketId, "maskedMessages": masked})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
