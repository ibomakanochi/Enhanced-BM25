import math
from collections import defaultdict

class EnhancedBM25Scorer:
    def __init__(self, corpus, preprocessor, k1=1.5, beta=0.5, tau=5.0, alpha=0.2, delta=0.5, epsilon=0.25):
        self.k1 = k1
        self.beta = beta
        self.tau = tau
        self.alpha = alpha
        self.delta = delta
        self.epsilon = epsilon
        self.corpus = corpus
        self.N = len(corpus)
        
        self.df = defaultdict(int)
        self.cdn_scores = {}
        # Inverted index: term -> list of (doc_id, tf, cooccur_score)
        self.inverted_index = defaultdict(list)
        
        total_length = 0
        doc_tokens_map = {}
        
        # Pass 1: Tokenize title + text and build corpus statistics
        for doc_id, doc in corpus.items():
            # Fix 1: Include Title + Text for a fair comparison with baseline
            full_text = doc.get("title", "") + " " + doc.get("text", "")
            tokens = preprocessor.clean(full_text)
            doc_tokens_map[doc_id] = tokens
            doc_len = len(tokens)
            total_length += doc_len
            
            # Document frequency
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] += 1

        self.avgdl = total_length / self.N if self.N > 0 else 1.0

        # Pass 2: Pre-compute CDN and inverted index with term cluster positions
        for doc_id, tokens in doc_tokens_map.items():
            length = len(tokens)
            
            # Fix 2: Pre-compute CDN once per document (SOP 3)
            if length == 0:
                self.cdn_scores[doc_id] = 1.0
            else:
                log_norm = math.log(1 + length) / math.log(1 + self.avgdl)
                richness = len(set(tokens)) / length
                density_norm = 1.0 - richness
                self.cdn_scores[doc_id] = self.beta * log_norm + (1 - self.beta) * density_norm

            # Group term positions in a single pass to eliminate redundant scans
            positions_by_term = defaultdict(list)
            for idx, term in enumerate(tokens):
                positions_by_term[term].append(idx)

            for term, positions in positions_by_term.items():
                tf = len(positions)
                # Fix 3: Calculate Term Clustering Density (Span) once
                if tf > 1:
                    span = (positions[-1] - positions[0]) + 1
                    cooccur_score = tf / span
                else:
                    cooccur_score = 0.0
                
                self.inverted_index[term].append((doc_id, tf, cooccur_score))

        # Free temporary token dictionary from memory
        del doc_tokens_map

    def _compute_idf(self, term):
        # SOP 2: Epsilon floor to prevent the IDF Bias Wall
        df_t = self.df.get(term, 0)
        idf = math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1)
        idf_floored = max(idf, self.epsilon)
        return min(idf_floored, self.tau)

    def retrieve(self, expanded_query, top_k=10):
        doc_scores = defaultdict(float)
        
        # Fix 4: Iterate ONLY over documents containing query terms via Inverted Index
        for term, weight in expanded_query.items():
            if term not in self.inverted_index:
                continue
                
            idf_cap = self._compute_idf(term)
            
            for doc_id, tf, cooccur_score in self.inverted_index[term]:
                cdn = self.cdn_scores[doc_id]
                dfc_idf = idf_cap * (1 + self.alpha * cooccur_score)
                
                # SOP 2 Fix: Apply BM25+ Delta shift directly to normalized TF
                tf_norm = ((tf * (self.k1 + 1)) / (tf + self.k1 * cdn)) + self.delta
                
                doc_scores[doc_id] += dfc_idf * tf_norm * weight

        # Sort and return top_k candidates
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(score, doc_id, self.corpus[doc_id].get('text', '')) for doc_id, score in ranked]