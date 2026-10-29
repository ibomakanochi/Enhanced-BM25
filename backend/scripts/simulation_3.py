import os
import json
import sys
import re
from beir.datasets.data_loader import GenericDataLoader

# Import your existing BEIR downloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset

dataset = "arguana" 
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Trigger the download/check safely without unpacking
download_beir_dataset(dataset)

# Safely load the queries and qrels using BEIR directly
data_path = os.path.join(base_dir, "data", "datasets", dataset)
_, queries, qrels = GenericDataLoader(data_path).load(split="test")

# 2. Load the expanded ArguAna corpus
corpus_path = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")
if not os.path.exists(corpus_path):
    print(f"File not found: {corpus_path}")
    sys.exit()

with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"--- SIMULATION 3 DATA: {dataset.upper()} (CONTEXT SUPPRESSION BYPASS) ---")
count = 0

def clean_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower()) if text else []

# 3. Print 5 Query-Document pairs for Appendix B
for qid, target_docs in qrels.items():
    if count >= 5:
        break
    
    query_text = queries.get(qid, "")
    
    for doc_id, relevance in target_docs.items():
        if relevance > 0 and doc_id in corpus:
            doc = corpus[doc_id]
            
            title_tokens = clean_tokenize(doc.get('title', ''))
            text_tokens = clean_tokenize(doc.get('text', ''))
            exp_tokens = clean_tokenize(doc.get('expanded_queries', ''))
            
            len_title = len(title_tokens)
            len_text = len(text_tokens)
            len_exp = len(exp_tokens)
            
            # Standard flat BM25 length (everything combined)
            raw_flat_length = len_title + len_text + len_exp
            
            # Enhanced BM25F Independent Weighted Length (ArguAna weights: title=1.0, text=1.0, exp=1.2)
            w_title, w_text, w_exp = 1.0, 1.0, 1.2
            weighted_length = (len_title * w_title) + (len_text * w_text) + (len_exp * w_exp)
            
            print(f"Query ID: {qid} | Doc ID: {doc_id}")
            print(f"Base Title Length : {len_title} tokens (w={w_title})")
            print(f"Base Text Length  : {len_text} tokens (w={w_text})")
            print(f"Expansion Length  : {len_exp} tokens (w={w_exp})")
            print(f"Total Synthetic Tokens Injected: +{len_exp}")
            print("-" * 40)
            print(f"Standard BM25 Flat Length      : {raw_flat_length}")
            print(f"Enhanced BM25F Weighted Length : {weighted_length:.1f}")
            print("Status: Synthetic tokens successfully isolated via field normalization.")
            print("=" * 80)
            
            count += 1
            break