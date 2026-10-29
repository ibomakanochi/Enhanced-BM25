import sys
import os
import json
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset
from core.preprocessor import TextPreprocessor
from scripts.evaluate_enhanced import EnhancedBM25F  # Imports your new class

def test_scifact_table_4_3():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = "scifact"
    
    # 1. Trigger the download/check safely
    download_beir_dataset(dataset)

    # Safely load the queries and qrels using BEIR directly
    data_path = os.path.join(base_dir, "data", "datasets", dataset)
    _, queries, qrels = GenericDataLoader(data_path).load(split="test")
    queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}
    
    # Load expanded corpus
    expanded_corpus_path = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")
    with open(expanded_corpus_path, "r") as f:
        corpus = json.load(f)
        
    preprocessor = TextPreprocessor()
    
    # The specific configurations for Table 4.3
    configs = [
        {"name": "Flat BM25 Equivalent", "w_title": 1.0, "w_text": 1.0, "w_exp": 0.0, "k1": 1.3, "b": 0.80},
        {"name": "Moderate Bias Override", "w_title": 2.0, "w_text": 1.0, "w_exp": 0.0, "k1": 1.3, "b": 0.80},
        {"name": "Maximum Bias Override", "w_title": 3.5, "w_text": 1.0, "w_exp": 0.0, "k1": 1.3, "b": 0.80}
    ]
    
    print("Starting targeted test for Table 4.3 (SciFact)...")
    
    for config in configs:
        w_title = config["w_title"]
        w_exp = config["w_exp"]
        k1 = config["k1"]
        b = config["b"]
        name = config["name"]
        
        # 1. Build Index
        bm25f = EnhancedBM25F(corpus, preprocessor, w_title=w_title, w_text=1.0, w_exp=w_exp, k1=k1, b=b)
        
        # 2. Retrieve
        results = {}
        for qid, qtext in queries.items():
            q_tokens = preprocessor.clean(qtext)
            top_docs = bm25f.get_top_k(q_tokens, k=10)
            results[qid] = {doc_id: score for doc_id, score in top_docs}
            
        # 3. Evaluate
        evaluator = EvaluateRetrieval()
        ndcg, _, _, _ = evaluator.evaluate(qrels, results, k_values=[10])
        current_ndcg = ndcg['NDCG@10']
        
        print(f"{name} (w_title={w_title}) => NDCG@10: {current_ndcg:.4f}")

if __name__ == "__main__":
    test_scifact_table_4_3()