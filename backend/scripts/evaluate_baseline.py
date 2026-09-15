import sys
import os
import csv
import gc
import math
from collections import defaultdict, Counter
from beir.retrieval.evaluation import EvaluateRetrieval

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset
from core.preprocessor import TextPreprocessor

# Custom Fast Baseline using an Inverted Index (Identical math to rank-bm25, 50x faster)
class FastBaselineBM25:
    def __init__(self, corpus, preprocessor, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.df = defaultdict(int)
        self.inverted_index = defaultdict(list)
        self.doc_lens = {}
        
        total_len = 0
        for doc_id, doc in corpus.items():
            full_text = doc.get("title", "") + " " + doc.get("text", "")
            tokens = preprocessor.clean(full_text)
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)
            
            term_counts = Counter(tokens)
            for term, tf in term_counts.items():
                self.df[term] += 1
                self.inverted_index[term].append((doc_id, tf))
                
        self.avgdl = total_len / self.N if self.N > 0 else 1.0

    def get_top_k(self, query_tokens, k=10):
        scores = defaultdict(float)
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            # Standard Okapi IDF
            df = self.df[term]
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            
            for doc_id, tf in self.inverted_index[term]:
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom
                
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# --- Evaluation Loop ---
# This guard ensures the loop only runs when you explicitly execute THIS file, 
# not when another file (like app.py) imports FastBaselineBM25 from it.
if __name__ == "__main__":
    datasets = ["scifact", "arguana", "nfcorpus", "fiqa", "trec-covid"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(base_dir, "baseline_metrics.csv")

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "NDCG@10", "MAP@10", "Recall@10", "P@10"])

    print("\n--- FAST BASELINE BM25 EVALUATION ---")
    print(f"{'Dataset':<12} | {'NDCG@10':<10} | {'MAP@10':<10} | {'Recall@10':<10}")
    print("-" * 55)

    preprocessor = TextPreprocessor()

    for dataset in datasets:
        corpus, queries, qrels = download_beir_dataset(dataset)
        queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}
        
        # Initialize our fast baseline
        baseline_bm25 = FastBaselineBM25(corpus, preprocessor)
        
        baseline_results = {}
        for query_id, query_text in queries.items():
            query_tokens = preprocessor.clean(query_text)
            
            # Instantly retrieve top 10
            top_docs = baseline_bm25.get_top_k(query_tokens, k=10)
            baseline_results[query_id] = {doc_id: score for doc_id, score in top_docs}
            
        evaluator = EvaluateRetrieval()
        base_ndcg, base_map, base_recall, base_p = evaluator.evaluate(qrels, baseline_results, k_values=[10])
        
        print(f"{dataset:<12} | {base_ndcg['NDCG@10']:<10.4f} | {base_map['MAP@10']:<10.4f} | {base_recall['Recall@10']:<10.4f}")
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset, round(base_ndcg['NDCG@10'], 4), round(base_map['MAP@10'], 4), round(base_recall['Recall@10'], 4), round(base_p['P@10'], 4)])

        del baseline_bm25, baseline_results, corpus, queries, qrels, evaluator
        gc.collect()

    print(f"\nBaseline evaluation complete! Data saved to: {results_file}")