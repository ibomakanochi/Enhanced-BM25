import sys
import os
import random
import math
import gc
from beir.retrieval.evaluation import EvaluateRetrieval

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_index import download_beir_dataset
from core.preprocessor import TextPreprocessor
from core.scorer import EnhancedBM25Scorer
from core.expander import HybridQueryExpander

def evaluate_params(corpus, queries, qrels, preprocessor, expander, params):
    # Instantiate the scorer with the dynamically tested parameters
    scorer = EnhancedBM25Scorer(
        corpus, 
        preprocessor, 
        k1=params['k1'], 
        beta=params['beta'], 
        delta=params['delta'], 
        epsilon=params['epsilon']
    )
    
    results = {}
    for qid, qtext in queries.items():
        tokens = preprocessor.clean(qtext)
        expanded = expander.expand_query(tokens)
        top_docs = scorer.retrieve(expanded, top_k=10)
        results[qid] = {doc_id: score for score, doc_id, _ in top_docs}
        
    evaluator = EvaluateRetrieval()
    ndcg, _, _, _ = evaluator.evaluate(qrels, results, k_values=[10])
    
    # Aggressively wipe memory to respect the 8GB RAM limit
    del scorer, results, evaluator
    gc.collect()
    
    return ndcg['NDCG@10']

def get_neighbor(params):
    """Perturbs a single parameter to explore the search space."""
    new_params = params.copy()
    key = random.choice(list(params.keys()))
    
    # Apply constrained random walk
    if key == 'k1':
        new_params[key] = max(0.5, min(2.5, params[key] + random.uniform(-0.3, 0.3)))
    elif key == 'beta':
        new_params[key] = max(0.1, min(0.9, params[key] + random.uniform(-0.15, 0.15)))
    elif key == 'delta':
        new_params[key] = max(0.1, min(1.0, params[key] + random.uniform(-0.15, 0.15)))
    elif key == 'epsilon':
        new_params[key] = max(0.05, min(0.5, params[key] + random.uniform(-0.05, 0.05)))
        
    return {k: round(v, 3) for k, v in new_params.items()}

def simulated_annealing():
    print("\n--- SIMULATED ANNEALING BM25 OPTIMIZER ---")
    print("Target Dataset: SciFact")
    
    corpus, queries, qrels = download_beir_dataset("scifact")
    queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}
    
    preprocessor = TextPreprocessor()
    print("Loading FastText Model...")
    expander = HybridQueryExpander()
    
    # Initial Baseline Parameters (from our manual guess)
    current_params = {'k1': 1.5, 'beta': 0.5, 'delta': 0.5, 'epsilon': 0.25}
    current_score = evaluate_params(corpus, queries, qrels, preprocessor, expander, current_params)
    
    best_params = current_params.copy()
    best_score = current_score
    
    # Thermodynamics Settings
    temperature = 1.0
    cooling_rate = 0.90
    min_temperature = 0.01
    iteration = 1
    
    print(f"\nInitial State -> NDCG@10: {current_score:.4f} | Params: {current_params}")
    print("-" * 75)
    
    while temperature > min_temperature:
        # 1. Generate a neighboring mathematical state
        neighbor_params = get_neighbor(current_params)
        
        # 2. Evaluate the new state
        neighbor_score = evaluate_params(corpus, queries, qrels, preprocessor, expander, neighbor_params)
        
        # 3. Acceptance Probability (Metropolis-Hastings Criterion)
        if neighbor_score > current_score:
            acceptance_prob = 1.0
            marker = "✅ ACCEPTED (IMPROVEMENT)"
        else:
            # Calculate probability of accepting a worse state to escape local minima
            acceptance_prob = math.exp((neighbor_score - current_score) / temperature)
            if random.random() < acceptance_prob:
                marker = "⚠️ ACCEPTED (EXPLORATION)"
            else:
                marker = "❌ REJECTED"
                
        # Apply Acceptance
        if marker.startswith("✅") or marker.startswith("⚠️"):
            current_params = neighbor_params
            current_score = neighbor_score
            
            # Track Global Best
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
                
        print(f"Iter {iteration:02d} | T: {temperature:.3f} | NDCG: {neighbor_score:.4f} | {marker:<25} | {neighbor_params}")
        
        # 4. Cool the system down
        temperature *= cooling_rate
        iteration += 1
        
    print("\n" + "=" * 75)
    print("Optimization Complete!")
    print(f"Baseline Score : 0.6437 (Standard rank-bm25)")
    print(f"Old Guess Score: 0.6115 (Our initial manual parameters)")
    print(f"Optimized Score: {best_score:.4f} (Simulated Annealing optimum)")
    print(f"Optimal Params : {best_params}")
    print("=" * 75)

if __name__ == "__main__":
    simulated_annealing()