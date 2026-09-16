# Enhanced Two-Stage Retrieval Architecture for RAG Systems

This repository contains the backend implementation of a Two-Stage Retrieval Engine designed for Retrieval-Augmented Generation (RAG) systems. The study enhances standard lexical retrieval by combining the speed of BM25 with the deep semantic understanding of a Neural Cross-Encoder, engineered to operate strictly within edge-hardware constraints (8GB RAM limit) without external APIs.

## ?? System Architecture Overview

The retrieval pipeline addresses the Lexical Wall, IDF Bias Wall, and Context Suppression Wall through a two-stage sequential architecture:
1. **Stage 1 (High-Recall Preliminary Filter):** A fast, in-memory BM25 index scores the entire text corpus and isolates the top `$K=25` candidates, protecting the system from memory exhaustion and bypassing baseline length penalties.
2. **Stage 2 (Deep Semantic Reranker):** A local MiniLM Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) concatenates the query and the 25 candidate documents, utilizing neural self-attention to score true contextual intent. The system then outputs the final top `$K` documents.

## ?? Repository Structure

The codebase is organized to strictly separate core algorithmic processing, offline benchmark evaluation, and the live RAG application.

    backend/
    +-- api/
    ¦   +-- app.py                  # Live RAG Flask Server (Integration with Llama-3)
    +-- core/
    ¦   +-- preprocessor.py         # Text cleaning and tokenization
    ¦   +-- reranker.py             # Neural Cross-Encoder (Stage 2)
    +-- scripts/
        +-- build_index.py          # Automated BEIR dataset ingestion
        +-- evaluate_baseline.py    # Offline Stage 1 testing (NDCG, MAP, Recall)
        +-- evaluate_enhanced.py    # Offline Two-Stage testing (NDCG, MAP, Recall)

## ?? System Requirements
* **Processor:** Intel Core i3 (2.0 GHz) Minimum / i5+ Recommended
* **Memory:** 8 GB RAM (Strict hardware boundary)
* **Python:** v3.8 or higher
* **LLM Engine:** Ollama (Required only for the live app.py server)

## ?? Installation & Setup

1. **Clone the repository and navigate to the root directory:**

    git clone <your-repository-url>
    cd <repository-name>

2. **Set up a virtual environment (Recommended):**

    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. **Install the required Python dependencies:**
*(Ensure you have flask, flask-cors, requests, beir, rank-bm25, and sentence-transformers installed).*

    pip install flask flask-cors requests beir rank-bm25 sentence-transformers

## ?? Running the Offline Evaluation Benchmarks
The evaluation scripts automatically download the required BEIR benchmark datasets (e.g., SciFact, TREC-COVID) and compute NDCG@10, MAP@10, and Recall@10. No internet is required after the initial dataset and model download.

**1. Run the Baseline BM25 Benchmark:**
Evaluates the standard, single-stage lexical approach.

    python backend/scripts/evaluate_baseline.py

*Outputs: baseline_metrics.csv*

**2. Run the Enhanced Two-Stage Benchmark:**
Evaluates the proposed pipeline (BM25 Top 25 -> Cross-Encoder Top 10).

    python backend/scripts/evaluate_enhanced.py

*Outputs: enhanced_metrics.csv*

## ?? Running the Live RAG Application
The app.py script launches the retrieval pipeline as a local web service and streams answers using a local Large Language Model (Llama-3).

**1. Start the local LLM via Ollama:**
Ensure Ollama is installed on your machine, open a separate terminal, and run:

    ollama run llama3:8b

**2. Start the Flask Backend:**
In your main terminal, start the Python API:

    python backend/api/app.py

*The server will initialize the SciFact corpus, load the MiniLM Cross-Encoder into memory, and listen for POST requests on [http://127.0.0.1:5000/api/retrieve_and_generate](http://127.0.0.1:5000/api/retrieve_and_generate).*

---
**Author:** Daniella 
