import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from tqdm import tqdm
from core.offline_expander import OfflineDocExpander
from scripts.build_index import download_beir_dataset

# Configuration for fault tolerance
CHECKPOINT_INTERVAL = 200

def expand_and_save_datasets():
    datasets = ["nfcorpus"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    expander = OfflineDocExpander()

    for dataset in datasets:
        print(f"\n--- Processing dataset: {dataset} ---")
        corpus, queries, qrels = download_beir_dataset(dataset)
        total_docs = len(corpus)
        print(f"Total documents in '{dataset}': {total_docs}")

        # Save directly to base_dir to match your unmodified evaluation script
        checkpoint_file = os.path.join(base_dir, f"checkpoint_{dataset}.json")
        final_output_file = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")

        processed_corpus = {}
        
        # 1. Load Checkpoint if it exists
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r") as f:
                    processed_corpus = json.load(f)
                print(f"Resumed from checkpoint. Already completed {len(processed_corpus)} / {total_docs} docs.")
            except Exception as e:
                print(f"Could not load checkpoint ({e}). Starting fresh.")
                processed_corpus = {}

        # 2. Filter documents that still need processing
        doc_items = [(doc_id, doc) for doc_id, doc in corpus.items() if doc_id not in processed_corpus]
        remaining_to_process = len(doc_items)
        print(f"Remaining documents to process: {remaining_to_process}")

        # 3. Process remaining documents with Checkpointing
        if remaining_to_process > 0:
            with tqdm(total=remaining_to_process, desc=f"Expanding {dataset}", unit="doc") as pbar:
                for step, (doc_id, doc) in enumerate(doc_items):
                    synthetic_queries = expander.generate_synthetic_queries(doc.get("text", ""), num_queries=3)
                    
                    doc["expanded_queries"] = synthetic_queries
                    processed_corpus[doc_id] = doc

                    completed_so_far = len(processed_corpus)
                    docs_left = total_docs - completed_so_far
                    
                    pbar.update(1)
                    pbar.set_postfix({"remaining": docs_left})

                    # Save intermediate progress locally
                    if (step + 1) % CHECKPOINT_INTERVAL == 0:
                        with open(checkpoint_file, "w") as f:
                            json.dump(processed_corpus, f)

            # 4. Save Final Output and Cleanup Checkpoint
            with open(final_output_file, "w") as f:
                json.dump(processed_corpus, f)
            
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)

        print(f"Successfully finished processing {dataset}!")
        print(f"Saved final corpus locally to: {final_output_file}")

if __name__ == "__main__":
    expand_and_save_datasets()