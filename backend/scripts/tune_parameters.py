import sys
import os
import json
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset
from core.preprocessor import TextPreprocessor
from scripts.evaluate_enhanced import EnhancedBM25F  # Imports your new class

def tune_scifact():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = "arguana"
    
    # Load raw queries and qrels
    _, queries, qrels = download_beir_dataset(dataset)
    queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}
    
    # Load expanded corpus
    expanded_corpus_path = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")
    with open(expanded_corpus_path, "r") as f:
        corpus = json.load(f)
        
    preprocessor = TextPreprocessor()
    
    # The Micro-Tuning Grid for SciFact
    title_weights = [1.0, 2.0, 3.0]
    exp_weights = [0.3, 0.7, 1.2]
    k1_values = [0.9, 1.2, 1.5]
    b_values = [0.4, 0.75]
    
    best_ndcg = 0
    best_params = {}
    
    print("Starting rapid parameter sweep for SciFact...")
    
    for w_title in title_weights:
        for w_exp in exp_weights:
            for k1 in k1_values:
                for b in b_values:
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
                    
                    print(f"w_title:{w_title} | w_exp:{w_exp} | k1:{k1} | b:{b}  =>  NDCG: {current_ndcg:.4f}")
                    
                    if current_ndcg > best_ndcg:
                        best_ndcg = current_ndcg
                        best_params = {'w_title': w_title, 'w_exp': w_exp, 'k1': k1, 'b': b}

    print("\n" + "="*50)
    print(f"WINNING CONFIGURATION FOR SCIFACT: NDCG@10 = {best_ndcg:.4f}")
    print(best_params)
    print("="*50)

if __name__ == "__main__":
    tune_scifact()