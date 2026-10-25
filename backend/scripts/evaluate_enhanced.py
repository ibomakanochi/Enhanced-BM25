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
        for doc_id, doc in tqdm(corpus.items(), desc="Indexing BM25F Corpus", unit="doc", leave=False):
            title_tokens = preprocessor.clean(doc.get("title", ""))
            text_tokens = preprocessor.clean(doc.get("text", ""))
            
            # In a real pipeline, this pulls from your offline Doc2Query generation.
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
    datasets = ["scifact", "nfcorpus", "arguana", "fiqa", "trec-covid", "scidocs", "webis-touche2020", "cqadupstack"]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(base_dir, "enhanced_metrics.csv")

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "NDCG@10", "MAP@10", "Recall@10", "P@10"])

    print("\n--- SINGLE-STAGE ENHANCED EVALUATION (BM25F + Doc2Query) ---")
    print(f"{'Dataset':<15} | {'NDCG@10':<10} | {'MAP@10':<10} | {'Recall@10':<10}")
    print("-" * 60)

    preprocessor = TextPreprocessor()
    
    configs = {
        "scifact":          {"w_title": 3.5, "w_text": 1.0, "w_exp": 0.0, "k1": 1.3, "b": 0.80},
        "nfcorpus":         {"w_title": 2.0, "w_text": 1.0, "w_exp": 1.2, "k1": 1.5, "b": 0.40},
        "arguana":          {"w_title": 1.0, "w_text": 1.0, "w_exp": 1.2, "k1": 1.5, "b": 0.75},
        "fiqa":             {"w_title": 1.0, "w_text": 1.0, "w_exp": 1.2, "k1": 1.5, "b": 0.40},
        "trec-covid":       {"w_title": 3.0, "w_text": 1.0, "w_exp": 1.2, "k1": 1.5, "b": 0.40},
        "scidocs":          {"w_title": 3.0, "w_text": 1.0, "w_exp": 0.3, "k1": 1.5, "b": 0.75},
        "webis-touche2020": {"w_title": 1.0, "w_text": 1.0, "w_exp": 0.3, "k1": 1.5, "b": 0.40},
        "cqadupstack":      {"w_title": 3.0, "w_text": 1.0, "w_exp": 0.7, "k1": 1.5, "b": 0.75},
        "default":          {"w_title": 1.0, "w_text": 1.0, "w_exp": 1.0, "k1": 1.5, "b": 0.75}
    }

    for dataset in datasets:
        # download_beir_dataset now returns a dict: {'android': (corpus, queries, qrels), ...} or {'scifact': (...)}
        loaded_data = download_beir_dataset(dataset)
        
        # Load your locally saved EXPANDED corpus once per dataset
        expanded_corpus_path = os.path.join(base_dir, f"expanded_{dataset}_corpus.json")
        if not os.path.exists(expanded_corpus_path):
            print(f"Error: Expanded corpus not found for {dataset}. Please run run_expansion.py first.")
            continue
            
        with open(expanded_corpus_path, "r") as f:
            full_expanded_corpus = json.load(f)
            
        params = configs.get(dataset, configs["default"])
        subset_metrics = []

        # Iterate through the subsets (1 for normal datasets, 12 for cqadupstack)
        for subset_name, (raw_corpus, queries, qrels) in loaded_data.items():
            queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}
            
            # Extract the correct corpus structure
            if dataset == "cqadupstack":
                subset_corpus = full_expanded_corpus.get(subset_name, {})
            else:
                subset_corpus = full_expanded_corpus
                
            enhanced_bm25f = EnhancedBM25F(
                subset_corpus, 
                preprocessor, 
                w_title=params["w_title"], 
                w_text=params["w_text"], 
                w_exp=params["w_exp"], 
                k1=params["k1"], 
                b=params["b"]
            )
            
            enhanced_results = {}
            for query_id, query_text in tqdm(queries.items(), desc=f"Evaluating {subset_name}", unit="q", leave=False):
                query_tokens = preprocessor.clean(query_text)
                top_docs = enhanced_bm25f.get_top_k(query_tokens, k=10)
                enhanced_results[query_id] = {doc_id: score for doc_id, score in top_docs}
                
            evaluator = EvaluateRetrieval()
            enh_ndcg, enh_map, enh_recall, enh_p = evaluator.evaluate(qrels, enhanced_results, k_values=[10])
            
            # Store metrics silently for macro-averaging
            subset_metrics.append({
                "ndcg": enh_ndcg['NDCG@10'],
                "map": enh_map['MAP@10'],
                "recall": enh_recall['Recall@10'],
                "p": enh_p['P@10']
            })
            
            del subset_corpus, queries, qrels, enhanced_bm25f, enhanced_results, evaluator
            gc.collect()

        # Calculate and print ONLY the macro-average for the entire dataset
        macro_ndcg = sum(m["ndcg"] for m in subset_metrics) / len(subset_metrics)
        macro_map = sum(m["map"] for m in subset_metrics) / len(subset_metrics)
        macro_recall = sum(m["recall"] for m in subset_metrics) / len(subset_metrics)
        macro_p = sum(m["p"] for m in subset_metrics) / len(subset_metrics)
        
        print(f"{dataset:<15} | {macro_ndcg:<10.4f} | {macro_map:<10.4f} | {macro_recall:<10.4f}")
        
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset, round(macro_ndcg, 4), round(macro_map, 4), round(macro_recall, 4), round(macro_p, 4)])

    print(f"\nEnhanced evaluation complete! Data saved to: {results_file}")