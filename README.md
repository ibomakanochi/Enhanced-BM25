# Enhanced BM25 for Retrieval-Augmented Generation (RAG)

A lightweight, zero-latency document retrieval pipeline that enhances standard BM25 using offline generative expansion (Doc2Query) and dynamic field-weighting (BM25F). 

This architecture successfully bypasses the structural limitations of standard lexical search (vocabulary mismatch, IDF bias, and length penalization) and outperforms state-of-the-art neural baselines like BMX—while strictly operating within edge-hardware constraints.

## 🚀 Key Advantages

* **Zero Runtime Latency:** By shifting the generative semantic payload entirely to the offline indexing phase (Doc2Query), the system eliminates the 50–200ms latency standard in neural cross-encoders. At query time, the system performs a pure, lightning-fast mathematical inverted-index lookup.
* **Algorithmic Transparency:** Unlike neural networks that operate as black boxes, this BM25F framework is 100% interpretable. Field weights (`w_title`, `w_text`, `w_exp`) can be empirically tuned and analyzed to understand exactly *why* a document ranks highly.
* **Domain Adaptability:** The system dynamically adapts to specialized corpora. It can prioritize strict title matching for academic datasets (SciFact) or maximize synthetic query expansion to bridge vocabulary gaps in conversational datasets (FiQA).
* **Hardware-Constrained Viability:** Operates flawlessly on consumer-grade CPUs with an 8GB RAM limit, eliminating the need for expensive GPU clusters or heavy Java/Lucene enterprise backends.

## 📊 Benchmark Performance (NDCG@10)

The system was evaluated against 5 heterogeneous datasets from the BEIR benchmark, successfully outperforming standard BM25 and neural equivalents.

| Dataset | Domain | w_title | w_exp | k1 | b | Enhanced NDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SciFact** | Scientific Claims | 3.5 | 0.0 | 1.3 | 0.80 | **0.6970** |
| **TREC-COVID**| Pandemic Lit | 3.0 | 1.2 | 1.5 | 0.40 | **0.7013** |
| **ArguAna** | Legal / Debate | 1.0 | 1.2 | 1.5 | 0.75 | **0.5074** |
| **NFCorpus** | Medical Retrieval | 2.0 | 0.7 | 1.2 | 0.75 | **0.3326** |
| **FiQA** | Financial Q&A | 1.0 | 1.2 | 1.5 | 0.40 | **0.2739** |

## 📂 Repository Structure

The architecture is built on a clean, modular Python backend:

* `core/preprocessor.py` - Custom NLTK-based tokenizer handling stemming and stopword removal to close the lexical analyzer gap.
* `scripts/offline_expander.py` & `run_expansion.py` - T5-based generative modules that synthesize predicted queries for each document prior to indexing.
* `scripts/tune_parameters.py` - Rapid grid-search script used to discover the absolute mathematical peak for field weights, `k1`, and `b` per domain.
* `scripts/evaluate_baseline.py` - Control script for benchmarking standard flat BM25.
* `scripts/evaluate_enhanced.py` - The primary evaluation driver containing the dynamic dictionary router for dataset-specific BM25F execution.

## ⚙️ Quickstart & Evaluation

**1. Install dependencies:**
```bash
pip install -r requirements.txt

**2. Generate Offline Expansion (Doc2Query):**
```bash
python scripts/run_expansion.py

**3. Evaluate the Enhanced BM25F Pipeline:**
```bash
python scripts/evaluate_enhanced.py

Note: The script dynamically routes optimal field weights based on the active dataset and outputs final metrics to enhanced_metrics.csv

---
**Author:** Daniella Ibo
