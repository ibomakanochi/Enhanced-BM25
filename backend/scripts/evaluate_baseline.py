import sys
import os
import csv
import gc
import math
from collections import defaultdict, Counter
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset
from core.preprocessor import TextPreprocessor

class FastBaselineBM25:
    def __init__(self, corpus, preprocessor, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.df = defaultdict(int)
        self.inverted_index = defaultdict(list)
        self.doc_lens = {}
        
        total_len = 0
        
        # Added tqdm loading bar for the indexing phase
        for doc_id, doc in tqdm(corpus.items(), desc="Indexing BM25 Corpus", unit="doc", leave=False):
            title = doc.get("title", "")
            text = doc.get("text", "")
            
            # Standard Flat BM25 formulation
            full_text = f"{title} {text}".strip()
            
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
            
            df = self.df[term]
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            
            for doc_id, tf in self.inverted_index[term]:
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom
                
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


if __name__ == "__main__":
    # Included all 8 datasets in the exact requested order
    datasets = [
        "scifact", "nfcorpus", "arguana", "fiqa", 
        "trec-covid", "scidocs", "webis-touche2020", "cqadupstack"
    ]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(base_dir, "baseline_metrics.csv")

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "NDCG@10", "MAP@10", "Recall@10", "P@10"])

    print("\n--- FAST BASELINE BM25 EVALUATION ---")
    print(f"{'Dataset':<15} | {'NDCG@10':<10} | {'MAP@10':<10} | {'Recall@10':<10}")
    print("-" * 60)

    preprocessor = TextPreprocessor()

    for dataset in datasets:
        # loaded_data contains {'android': (corpus, queries, qrels), ...} or {'scifact': ...}
        loaded_data = download_beir_dataset(dataset)
        
        subset_metrics = []

        for subset_name, (corpus, queries, qrels) in loaded_data.items():
            queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}
            
            baseline_bm25 = FastBaselineBM25(corpus, preprocessor)
            
            baseline_results = {}
            
            # Added tqdm loading bar for the query evaluation phase
            for query_id, query_text in tqdm(queries.items(), desc=f"Evaluating {subset_name}", unit="q", leave=False):
                query_tokens = preprocessor.clean(query_text)
                top_docs = baseline_bm25.get_top_k(query_tokens, k=10)
                baseline_results[query_id] = {doc_id: score for doc_id, score in top_docs}
                
            evaluator = EvaluateRetrieval()
            base_ndcg, base_map, base_recall, base_p = evaluator.evaluate(qrels, baseline_results, k_values=[10])
            
            # Store metrics silently for macro-averaging
            subset_metrics.append({
                "ndcg": base_ndcg['NDCG@10'],
                "map": base_map['MAP@10'],
                "recall": base_recall['Recall@10'],
                "p": base_p['P@10']
            })
            
            del baseline_bm25, baseline_results, corpus, queries, qrels, evaluator
            gc.collect()

        # Calculate and print ONLY the macro-average for the entire dataset
        macro_ndcg = sum(m["ndcg"] for m in subset_metrics) / len(subset_metrics)
        macro_map = sum(m["map"] for m in subset_metrics) / len(subset_metrics)
        macro_recall = sum(m["recall"] for m in subset_metrics) / len(subset_metrics)
        macro_p = sum(m["p"] for m in subset_metrics) / len(subset_metrics)
        
        # \r ensures that it overwrites the cleared tqdm bar smoothly
        print(f"\r{dataset:<15} | {macro_ndcg:<10.4f} | {macro_map:<10.4f} | {macro_recall:<10.4f}")
        
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset, round(macro_ndcg, 4), round(macro_map, 4), round(macro_recall, 4), round(macro_p, 4)])

    print(f"\nBaseline evaluation complete! Data saved to: {results_file}")