from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import sys
import os
import json

# Adjust path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.preprocessor import TextPreprocessor
from core.reranker import NeuralReranker
from scripts.evaluate_baseline import FastBaselineBM25
from scripts.build_index import download_beir_dataset

app = Flask(__name__)
CORS(app)

print("Bootstrapping Two-Stage Retrieval Engine...")
preprocessor = TextPreprocessor()

print("Loading SciFact Corpus...")

# We only need the corpus for the live app. (Queries and qrels are for evaluation). Change Dataset if said to do so.
scifact_corpus, _, _ = download_beir_dataset("scifact")

print("Initializing Fast BM25 Index (This will take a few seconds)...")
bm25_stage1 = FastBaselineBM25(scifact_corpus, preprocessor)

print("Loading Neural Cross-Encoder (MiniLM)...")
reranker = NeuralReranker()

@app.route('/api/retrieve_and_generate', methods=['POST'])
def retrieve_and_generate():
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    def generate_stream():
        # 1. RAG Retrieval Phase (Two-Stage Architecture)
        query_tokens = preprocessor.clean(user_query)
        
        # Stage 1: High Recall Retrieval (Fetch top 25 candidates from all 5,183 docs)
        top_k_stage1 = bm25_stage1.get_top_k(query_tokens, k=25)
        
        candidate_docs = [
            (score, doc_id, scifact_corpus[doc_id].get("title", "") + " " + scifact_corpus[doc_id].get("text", ""))
            for doc_id, score in top_k_stage1
        ]
        
        # Stage 2: High Precision Neural Reranking (Refine down to the Top 5 best context docs)
        top_docs = reranker.rerank(user_query, candidate_docs, top_k=5)
        
        retrieved_context = [{"score": doc[0], "text": doc[2]} for doc in top_docs]
        context_str = "\n\n".join([f"Doc {i+1}: {doc[2]}" for i, doc in enumerate(top_docs)])
        
        # System prompt instructing the LLM to use the retrieved scientific context
        prompt = f"You are a scientific research assistant. Use the following retrieved context to answer the question. If the context does not contain the answer, say so.\n\nContext:\n{context_str}\n\nQuestion: {user_query}\nAnswer:"
        
        # Send initial context back to the frontend immediately
        yield f"data: {json.dumps({'type': 'context', 'retrieved': retrieved_context})}\n\n"

        # 2. LLM Generation Phase (Streaming)
        try:
            response = requests.post(
                'http://localhost:11434/api/generate', 
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get('response', '')
                    is_done = chunk.get('done', False)
                    
                    if token:
                        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                        
                    if is_done:
                        break
                        
        except Exception as e:
            error_msg = f"Ollama Connection Error: Is Ollama running? Details: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'text': error_msg})}\n\n"

        # 3. Termination Signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)