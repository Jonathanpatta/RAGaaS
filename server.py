from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import time

app = Flask(__name__)
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

@app.route('/embed', methods=['POST'])
def embed_chunks():
    chunks = request.json['chunks']
    embeddings = model.encode(chunks)
    return jsonify({'embeddings': embeddings.tolist()})

@app.route('/embed_sentence', methods=['POST'])
def embed_sentence():
    sentence = request.json['sentence']
    embeddings = model.encode(sentences=[sentence])
    return jsonify({'embedding': embeddings[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)