import os
import json
import sys
from collections import Counter
import re

# Import your existing BEIR downloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset

dataset = "scifact" 
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Fetch the queries and qrels
_, queries, qrels = download_beir_dataset(dataset)

# 2. Load the expanded SciFact corpus
corpus_path = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"--- SIMULATION 2 DATA: {dataset.upper()} (IDF BIAS OVERRIDE) ---")
count = 0

def clean_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# 3. Print 5 Query-Document pairs for Appendix B
for qid, target_docs in qrels.items():
    if count >= 5:
        break
    
    query_text = queries.get(qid, "")
    query_tokens = clean_tokenize(query_text)
    
    for doc_id, relevance in target_docs.items():
        if relevance > 0 and doc_id in corpus:
            doc = corpus[doc_id]
            title_text = doc.get('title', '')
            body_text = doc.get('text', '')
            
            title_tokens = clean_tokenize(title_text)
            body_tokens = clean_tokenize(body_text)
            
            title_counts = Counter(title_tokens)
            body_counts = Counter(body_tokens)
            
            # Find overlapping terms
            matched_terms = [t for t in query_tokens if t in title_counts or t in body_counts]
            
            if not matched_terms:
                continue

            print(f"Query ID: {qid} | Doc ID: {doc_id}")
            print(f"User Query: {query_text}")
            print(f"Base Title: {title_text}")
            
            print("\nTerm Frequency & Structural Weighting Analysis (w_title=3.5, w_text=1.0):")
            for term in set(matched_terms):
                tf_title = title_counts[term]
                tf_text = body_counts[term]
                weighted_score = (tf_title * 3.5) + (tf_text * 1.0)
                print(f"  -> Term '{term}': Title TF = {tf_title} (x3.5), Body TF = {tf_text} (x1.0) | Weighted TF = {weighted_score}")
            
            print("-" * 80)
            count += 1
            break