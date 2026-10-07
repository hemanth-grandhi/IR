import heapq

class Ranker:
    def __init__(self, tfidf_scorer, title_weight=0.6, body_weight=0.4):
        self.scorer = tfidf_scorer
        self.title_weight = title_weight
        self.body_weight = body_weight
        
    def score_document(self, query_vec, query_norm, doc_id):
        """Computes cosine similarity for a single document, with zone weighting."""
        doc_data = self.scorer.doc_vectors.get(doc_id)
        if not doc_data:
            return 0.0, {}
            
        title_vec = doc_data['title_vec']
        body_vec = doc_data['body_vec']
        title_norm = doc_data['title_norm']
        body_norm = doc_data['body_norm']
        
        title_dot = 0.0
        body_dot = 0.0
        
        term_contributions = {} # For explainability
        
        for term, q_w in query_vec.items():
            # Title contribution
            t_w = title_vec.get(term, 0.0)
            t_dot = q_w * t_w
            title_dot += t_dot
            
            # Body contribution
            b_w = body_vec.get(term, 0.0)
            b_dot = q_w * b_w
            body_dot += b_dot
            
            if t_dot > 0 or b_dot > 0:
                term_contributions[term] = {
                    'title_score': t_dot / (query_norm * title_norm) if (query_norm * title_norm) > 0 else 0,
                    'body_score': b_dot / (query_norm * body_norm) if (query_norm * body_norm) > 0 else 0
                }
                
        # Cosine similarities
        title_sim = title_dot / (query_norm * title_norm) if (query_norm * title_norm) > 0 else 0.0
        body_sim = body_dot / (query_norm * body_norm) if (query_norm * body_norm) > 0 else 0.0
        
        # Zone weighting
        final_score = self.title_weight * title_sim + self.body_weight * body_sim
        
        # Consolidate contributions
        for term in term_contributions:
            term_contributions[term]['total'] = (
                self.title_weight * term_contributions[term]['title_score'] +
                self.body_weight * term_contributions[term]['body_score']
            )
            
        return final_score, term_contributions
        
    def get_top_k(self, query_tokens, k=10):
        """Retrieves Top-K documents using Cosine Similarity and a Min-Heap (via heapq.nlargest)."""
        query_vec, query_norm = self.scorer.get_query_vector(query_tokens)
        
        if query_norm == 0:
            return [], {}
            
        scores = []
        all_contributions = {}
        
        # Get candidate documents (only docs that contain at least one query term)
        candidate_docs = set()
        for term in query_vec:
            if term in self.scorer.index.index:
                candidate_docs.update(self.scorer.index.index[term].keys())
                
        for doc_id in candidate_docs:
            score, contributions = self.score_document(query_vec, query_norm, doc_id)
            if score > 0:
                scores.append((score, doc_id))
                all_contributions[doc_id] = contributions
                
        # Efficient Top-K using a heap (O(N log K) instead of O(N log N) sort)
        top_k = heapq.nlargest(k, scores, key=lambda x: x[0])
        
        return top_k, all_contributions
