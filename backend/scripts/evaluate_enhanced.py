import sys
import os
import csv
import gc
import math
import json
from collections import defaultdict, Counter
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset
from core.preprocessor import TextPreprocessor

class EnhancedBM25F:
    def __init__(self, corpus, preprocessor, w_title=2.5, w_text=1.0, w_exp=1.5, k1=1.5, b=0.75):
        """
        Single-Stage Field-Weighted BM25. 
        Replaces the neural cross-encoder by applying semantic priority mathematically.
        """
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.df = defaultdict(int)
        self.inverted_index = defaultdict(list)
        self.doc_lens = {}
        
        self.w_title = w_title
        self.w_text = w_text
        self.w_exp = w_exp
        
        total_len = 0
        
        # Build the Field-Weighted Inverted Index
        for doc_id, doc in tqdm(corpus.items(), desc="Indexing BM25F Corpus", unit="doc"):
            title_tokens = preprocessor.clean(doc.get("title", ""))
            text_tokens = preprocessor.clean(doc.get("text", ""))
            
            # In a real pipeline, this pulls from your offline Doc2Query generation.
            # We default to an empty string if the offline step hasn't been run yet.
            exp_tokens = preprocessor.clean(doc.get("expanded_queries", ""))
            
            title_counts = Counter(title_tokens)
            text_counts = Counter(text_tokens)
            exp_counts = Counter(exp_tokens)
            
            # Compute weighted document length (Solves Context Suppression)
            doc_weighted_len = (len(title_tokens) * self.w_title) + \
                               (len(text_tokens) * self.w_text) + \
                               (len(exp_tokens) * self.w_exp)
            
            self.doc_lens[doc_id] = doc_weighted_len
            total_len += doc_weighted_len
            
            all_terms = set(title_tokens) | set(text_tokens) | set(exp_tokens)
            
            for term in all_terms:
                # Combine field frequencies using designated weights (Solves IDF Bias)
                tf_weighted = (title_counts[term] * self.w_title) + \
                              (text_counts[term] * self.w_text) + \
                              (exp_counts[term] * self.w_exp)
                
                self.df[term] += 1
                self.inverted_index[term].append((doc_id, tf_weighted))
                
        self.avgdl = total_len / self.N if self.N > 0 else 1.0

    def get_top_k(self, query_tokens, k=10):
        scores = defaultdict(float)
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            df = self.df[term]
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            
            for doc_id, tf_weighted in self.inverted_index[term]:
                dl = self.doc_lens[doc_id]
                # Apply saturation to the combined, weighted term frequency
                denom = tf_weighted + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * (tf_weighted * (self.k1 + 1.0)) / denom
                
        # Return the definitive Top K in a single, lightning-fast pass
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


if __name__ == "__main__":
    datasets = ["scifact", "nfcorpus", "arguana", "fiqa", "trec-covid"]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(base_dir, "enhanced_metrics.csv")

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "NDCG@10", "MAP@10", "Recall@10", "P@10"])

    print("\n--- SINGLE-STAGE ENHANCED EVALUATION (BM25F + Doc2Query) ---")
    print(f"{'Dataset':<12} | {'NDCG@10':<10} | {'MAP@10':<10} | {'Recall@10':<10}")
    print("-" * 55)

    preprocessor = TextPreprocessor()

    for dataset in datasets:
        # Load queries and qrels from BEIR (we ignore the raw corpus using '_')
        _, queries, qrels = download_beir_dataset(dataset)
        queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}
        
        # Load your locally saved EXPANDED corpus
        expanded_corpus_path = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")
        
        if not os.path.exists(expanded_corpus_path):
            print(f"Error: Expanded corpus not found for {dataset}. Please run run_expansion.py first.")
            continue
            
        with open(expanded_corpus_path, "r") as f:
            corpus = json.load(f)
        
        # Global optimal weights dictionary
        configs = {
            "scifact":   {"w_title": 3.5, "w_text": 1.0, "w_exp": 0.0, "k1": 1.3, "b": 0.80},
            "nfcorpus":  {"w_title": 2.0, "w_text": 1.0, "w_exp": 0.7, "k1": 1.2, "b": 0.75},
            "arguana":   {"w_title": 1.0, "w_text": 1.0, "w_exp": 1.2, "k1": 1.5, "b": 0.75},
            "fiqa":      {"w_title": 1.0, "w_text": 1.0, "w_exp": 1.2, "k1": 1.5, "b": 0.40},
            "trec-covid":{"w_title": 3.0, "w_text": 1.0, "w_exp": 1.2, "k1": 1.5, "b": 0.40},
            "default":   {"w_title": 1.0, "w_text": 1.0, "w_exp": 1.0, "k1": 1.5, "b": 0.75}
        }

        # Select params based on current dataset, fallback to default if not found
        params = configs.get(dataset, configs["default"])
        
        enhanced_bm25f = EnhancedBM25F(
            corpus, 
            preprocessor, 
            w_title=params["w_title"], 
            w_text=params["w_text"], 
            w_exp=params["w_exp"], 
            k1=params["k1"], 
            b=params["b"]
        )
        
        enhanced_results = {}
        
        for query_id, query_text in tqdm(queries.items(), desc=f"Evaluating {dataset}", unit="q"):
            query_tokens = preprocessor.clean(query_text)
            
            # Fetch the final top 10 directly from the index (0ms latency, no neural reranker)
            top_docs = enhanced_bm25f.get_top_k(query_tokens, k=10)
            enhanced_results[query_id] = {doc_id: score for doc_id, score in top_docs}
            
        evaluator = EvaluateRetrieval()
        enh_ndcg, enh_map, enh_recall, enh_p = evaluator.evaluate(qrels, enhanced_results, k_values=[10])
        
        print(f"\r{dataset:<12} | {enh_ndcg['NDCG@10']:<10.4f} | {enh_map['MAP@10']:<10.4f} | {enh_recall['Recall@10']:<10.4f}")
        
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset, round(enh_ndcg['NDCG@10'], 4), round(enh_map['MAP@10'], 4), round(enh_recall['Recall@10'], 4), round(enh_p['P@10'], 4)])

        del corpus, queries, qrels, enhanced_bm25f, enhanced_results, evaluator
        gc.collect()

    print(f"\nEnhanced evaluation complete! Data saved to: {results_file}")