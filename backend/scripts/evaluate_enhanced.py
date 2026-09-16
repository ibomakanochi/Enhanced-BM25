import sys
import os
import csv
import gc
from beir.retrieval.evaluation import EvaluateRetrieval

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset
from core.preprocessor import TextPreprocessor
from core.reranker import NeuralReranker
from scripts.evaluate_baseline import FastBaselineBM25  # Reuse the fast inverted index

datasets = ["scifact", "arguana", "nfcorpus", "fiqa", "trec-covid"]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_file = os.path.join(base_dir, "enhanced_metrics.csv")

with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "NDCG@10", "MAP@10", "Recall@10", "P@10"])

print("\n--- TWO-STAGE ENHANCED EVALUATION (BM25 + Cross-Encoder) ---")
print(f"{'Dataset':<12} | {'NDCG@10':<10} | {'MAP@10':<10} | {'Recall@10':<10}")
print("-" * 55)

preprocessor = TextPreprocessor()
print("Loading Neural Cross-Encoder (MiniLM) into memory...")
reranker = NeuralReranker()

for dataset in datasets:
    corpus, queries, qrels = download_beir_dataset(dataset)
    queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}
    
    # Stage 1: Fast initial retrieval using standard BM25
    bm25_stage1 = FastBaselineBM25(corpus, preprocessor)
    
    enhanced_results = {}
    for query_id, query_text in queries.items():
        query_tokens = preprocessor.clean(query_text)
        
        # Fetch Top 25 candidates using BM25
        top_25_ids = bm25_stage1.get_top_k(query_tokens, k=25)
        
        # Prepare candidate objects for the reranker
        candidate_docs = [
            (score, doc_id, corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", ""))
            for doc_id, score in top_25_ids
        ]
        
        # Stage 2: Rerank the Top 25 down to the perfect Top 10 using the Neural Model
        top_10_reranked = reranker.rerank(query_text, candidate_docs, top_k=10)
        
        enhanced_results[query_id] = {doc_id: score for score, doc_id, _ in top_10_reranked}
        
    evaluator = EvaluateRetrieval()
    enh_ndcg, enh_map, enh_recall, enh_p = evaluator.evaluate(qrels, enhanced_results, k_values=[10])
    
    print(f"{dataset:<12} | {enh_ndcg['NDCG@10']:<10.4f} | {enh_map['MAP@10']:<10.4f} | {enh_recall['Recall@10']:<10.4f}")
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([dataset, round(enh_ndcg['NDCG@10'], 4), round(enh_map['MAP@10'], 4), round(enh_recall['Recall@10'], 4), round(enh_p['P@10'], 4)])

    del corpus, queries, qrels, bm25_stage1, enhanced_results, evaluator
    gc.collect()

print(f"\nEnhanced evaluation complete! Data saved to: {results_file}")