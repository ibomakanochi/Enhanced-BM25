import os
import json
from tqdm import tqdm
from core.offline_expander import OfflineDocExpander
from scripts.build_index import download_beir_dataset

def expand_and_save_datasets():
    datasets = ["scifact", "arguana", "nfcorpus", "fiqa", "trec-covid"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expander = OfflineDocExpander()

    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        corpus, queries, qrels = download_beir_dataset(dataset)
        
        # We only need to expand documents that are actually going to be searched
        for doc_id, doc in tqdm(corpus.items(), desc=f"Expanding {dataset}", unit="doc"):
            # Generate 3 synthetic queries per document
            synthetic_queries = expander.generate_synthetic_queries(doc.get("text", ""), num_queries=3)
            corpus[doc_id]["expanded_queries"] = synthetic_queries
            
        # Save the expanded corpus locally
        output_path = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")
        with open(output_path, "w") as f:
            json.dump(corpus, f)
        print(f"Saved expanded corpus to {output_path}")

if __name__ == "__main__":
    expand_and_save_datasets()