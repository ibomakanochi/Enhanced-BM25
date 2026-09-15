# Enhanced Two-Stage RAG System: BM25 + Neural Cross-Encoder

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-18-61dafb.svg)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey.svg)
![Ollama](https://img.shields.io/badge/Ollama-Llama--3-black.svg)

This repository contains the source code for a BSCS thesis project demonstrating an enhanced **Retrieval-Augmented Generation (RAG)** architecture. 

It proposes a **Two-Stage Neural Retrieval Pipeline** designed to overcome the structural limitations of standard BM25—specifically the *Lexical Wall*, *IDF Bias Wall*, and *Context Suppression Wall*—while remaining highly optimized for edge-first, hardware-constrained environments (e.g., systems limited to 8GB of RAM).

---

## 🏗️ System Architecture

The pipeline operates in three sequential phases:

### 1. Stage 1: High-Recall Lexical Filter (Fast BM25)
A highly optimized, in-memory inverted index that rapidly filters corpora containing hundreds of thousands of scientific documents (such as TREC-COVID or SciFact). Instead of passing results directly to an LLM, this stage acts as a computational funnel, narrowing the search space down to the top k=25 candidates based on exact keyword overlap.

### 2. Stage 2: High-Precision Semantic Reranker (Neural Cross-Encoder)
A MiniLM neural network evaluates the top 25 candidates. By reading the user query and the document simultaneously through attention layers, the Cross-Encoder computes deep semantic relevance. This resolves BM25's lexical rigidity and isolates the **Top 5** most contextually rich documents.

### 3. Stage 3: Edge-First LLM Generation (Streaming)
The Top 5 documents are concatenated into a strict system prompt to ground the downstream language model. The system utilizes a local **Llama-3 (8B)** model running via Ollama to ensure complete data privacy. The frontend consumes the generated output via Server-Sent Events (SSE), streaming the response token-by-token for a frictionless user experience.

---

## 🚀 Tech Stack

* **Backend:** Python, Flask, HuggingFace (sentence-transformers)
* **Frontend:** React.js, Server-Sent Events (SSE)
* **LLM Engine:** Ollama (Llama-3 8B)
* **Evaluation Datasets:** BEIR Benchmark (SciFact, TREC-COVID, NFCorpus, FiQA, Arguana)

---

## 🛠️ Installation & Local Setup

### Prerequisites
* **Python 3.9+**
* **Node.js (v16+)**
* **[Ollama](https://ollama.com/)** installed and running locally with the Llama-3 model pulled (ollama run llama3:8b).

### 1. Backend Setup
Navigate to the backend directory, set up the Python virtual environment, and install dependencies:

 `bash
cd backend
python -m venv venv

# Activate Virtual Environment (Windows)
venv\Scripts\activate
# Activate Virtual Environment (Mac/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
 `

Start the Flask API server. On the first run, it will automatically download the BEIR dataset and build the inverted index.
 `bash
python api/app.py
 `
*The server will run on http://127.0.0.1:5000.*

### 2. Frontend Setup
Open a new terminal window, navigate to the frontend directory, install packages, and start the React app:

 `bash
cd frontend
npm install
npm start
 `
*The UI will open automatically in your browser at http://localhost:3000.*

---

## 🧪 Thesis Benchmarking & Evaluation

To run the offline baseline evaluations (measuring NDCG@10, MAP@10, and Recall@10) across the BEIR datasets without starting the web server, run the dedicated evaluation script:

 `bash
# Ensure your backend virtual environment is activated
python scripts/evaluate_baseline.py
 `
*Results are automatically saved and appended to baseline_metrics.csv.*
