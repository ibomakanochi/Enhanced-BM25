import os
import json
import sys
from beir.datasets.data_loader import GenericDataLoader

# Import your existing BEIR downloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset

dataset = "nfcorpus" 
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Trigger the download/check, but don't try to unpack its return value
download_beir_dataset(dataset)

# Safely load the queries and qrels using BEIR directly
data_path = os.path.join(base_dir, "data", "datasets", dataset)
_, queries, qrels = GenericDataLoader(data_path).load(split="test")

# 2. Load your expanded corpus
corpus_path = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"--- SIMULATION DATA: {dataset.upper()} ---")
count = 0

# 3. Print 5 Query-Document pairs for Appendix B
for qid, target_docs in qrels.items():
    if count >= 5:
        break
    
    query_text = queries.get(qid, "Query not found")
    
    # Get the first relevant document for this query
    for doc_id, relevance in target_docs.items():
        if relevance > 0 and doc_id in corpus:
            doc = corpus[doc_id]
            print(f"Query ID: {qid} | Doc ID: {doc_id}")
            print(f"User Query: {query_text}")
            print(f"Base Title: {doc.get('title', '')}")
            # Truncated for readability in terminal
            print(f"Base Text Snippet: {doc.get('text', '')[:150]}...")
            print(f"Generative Expansion (exp): {doc.get('expanded_queries', '')[:150]}...\n")
            print("-" * 80)
            count += 1
            break # Move to the next query