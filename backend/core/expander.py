import json
import os
import fasttext

class HybridQueryExpander:
    def __init__(self, pmi_table_path="backend/data/pmi_table.json", ft_model_path="backend/models/cc.en.300.bin", gamma=0.5, theta=0.3, min_ft_sim=0.75, penalty_weight=0.15):
        self.gamma = gamma                # SOP 1: Penalty multiplier for PMI synonyms
        self.theta = theta                # SOP 1: Minimum PMI score threshold
        self.min_ft_sim = min_ft_sim      # SOP 1: Strict similarity threshold for FastText
        self.penalty_weight = penalty_weight # SOP 1: Penalty multiplier for FastText synonyms
        
        # Load O(1) NPMI lookup table
        if os.path.exists(pmi_table_path):
            with open(pmi_table_path, 'r') as f:
                self.pmi_table = json.load(f)
        else:
            self.pmi_table = {}

        # Load FastText semantic fallback
        if os.path.exists(ft_model_path):
            try:
                self.ft_model = fasttext.load_model(ft_model_path)
            except Exception:
                self.ft_model = None
        else:
            self.ft_model = None

    def expand_query(self, query_tokens):
        expanded = {}
        for token in query_tokens:
            if not token:
                continue
                
            # Original term retains maximum authority (Weight = 1.0)
            expanded[token] = expanded.get(token, 0) + 1.0  
            
            # SOP 1 Fix: Skip expanding short words, verbs, or numbers to prevent drift
            if len(token) <= 4 or token.isnumeric():
                continue
            
            expanded_via_pmi = False
            
            # Stage 1: PMI Primary Stage
            if token in self.pmi_table and self.pmi_table[token].get('score', 0) >= self.theta:
                for syn, score in self.pmi_table[token].get('synonyms', {}).items():
                    # Apply gamma penalty so PMI synonyms act as tie-breakers, not replacements
                    weighted_score = round(float(score) * self.gamma, 4)
                    if syn not in expanded or expanded[syn] < weighted_score:
                        expanded[syn] = weighted_score
                expanded_via_pmi = True
            
            # Stage 2: FastText Fallback Stage (Only runs if PMI failed)
            elif self.ft_model:
                try:
                    # Fetch top 2 nearest neighbors
                    neighbors = self.ft_model.get_nearest_neighbors(token, k=2)
                    for sim, neighbor in neighbors:
                        clean_neighbor = neighbor.strip().lower()
                        
                        # SOP 1 Fix: Enforce strict similarity boundary (>= 0.75)
                        if sim >= self.min_ft_sim:
                            weighted_score = round(float(sim) * self.penalty_weight, 4)
                            if clean_neighbor not in expanded or expanded[clean_neighbor] < weighted_score:
                                expanded[clean_neighbor] = weighted_score
                except Exception:
                    pass
                    
        return expanded