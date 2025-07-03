from flask import Flask, request, jsonify, Response
import json
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

        def generate_embeddings():
            for item in batches:
                ticketId = item.get("ticketId")
                message = item.get("message", "")
                embedding = get_embedding(message)
                result = {
                    "ticketId": ticketId,
                    "embedding": embedding
                }
                yield json.dumps(result) + '\n'

        return Response(generate_embeddings(), 
                       mimetype='application/x-ndjson',
                       headers={'Content-Type': 'application/x-ndjson'})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/mask_batch', methods=['POST'])
def mask_batch():
    try:
        data = request.get_json()
        batches = data.get('batches', [])  

        def generate_masked_results():
            for item in batches:
                ticketId = item.get("ticketId")
                messages = item.get("messages", [])
                masked = [full_masking_pipeline(m) for m in messages if m.strip()]
                result = {
                    "ticketId": ticketId, 
                    "maskedMessages": masked
                }
                
                # Convert to JSON and check size
                json_str = json.dumps(result)
                json_size = len(json_str.encode('utf-8'))
                
                # Log large responses
                if json_size > 200000:  # > 200KB
                    print(f"WARNING: Large response for ticket {ticketId}: {json_size} bytes")
                    print(f"Number of messages: {len(masked)}")
                    print(f"Total text length: {sum(len(msg) for msg in masked)}")
                
                # If response is too large, split it
                if json_size > 250000:  # > 250KB, split into chunks
                    print(f"Splitting large response for ticket {ticketId}")
                    chunk_size = 50  # messages per chunk
                    for i in range(0, len(masked), chunk_size):
                        chunk_messages = masked[i:i + chunk_size]
                        chunk_result = {
                            "ticketId": f"{ticketId}_chunk_{i//chunk_size}",
                            "maskedMessages": chunk_messages
                        }
                        yield json.dumps(chunk_result) + '\n'
                else:
                    yield json_str + '\n'

        return Response(generate_masked_results(), 
                       mimetype='application/x-ndjson',
                       headers={'Content-Type': 'application/x-ndjson'})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)