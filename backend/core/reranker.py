from sentence_transformers import CrossEncoder

class NeuralReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Loads a lightweight Cross-Encoder model locally
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query_text, candidate_docs, top_k=10):
        """
        Takes a query and a list of candidate documents (score, doc_id, text)
        and re-scores them using deep semantic cross-attention.
        """
        if not candidate_docs:
            return []
        
        # Format pairs for the Cross-Encoder: [[query, doc_text], [query, doc_text], ...]
        pairs = [[query_text, doc_tuple[2]] for doc_tuple in candidate_docs]
        
        # Predict semantic relevance scores
        semantic_scores = self.model.predict(pairs, show_progress_bar=True)
        
        # Attach the new neural scores to the documents
        reranked = [
            (float(semantic_scores[i]), candidate_docs[i][1], candidate_docs[i][2])
            for i in range(len(candidate_docs))
        ]
        
        # Sort by the new semantic score
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]