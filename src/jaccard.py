class JaccardBaseline:
    """A baseline using Jaccard coefficient for ranking documents."""
    def __init__(self, inverted_index, preprocessor):
        self.index = inverted_index
        self.preprocessor = preprocessor

    def get_top_k(self, query_tokens, k=10):
        if not query_tokens:
            return [], {}
            
        # Treat query as a set of tokens
        query_set = set(query_tokens)
        
        # We need the set of tokens for each candidate document
        candidate_docs = set()
        for term in query_set:
            postings = self.index.get_postings(term)
            candidate_docs.update(postings.keys())
            
        scores = {}
        for doc_id in candidate_docs:
            doc_tokens = set()
            metadata = self.index.doc_metadata.get(doc_id)
            if metadata:
                title = metadata.get('title', '')
                body = metadata.get('body', '')
                doc_tokens.update(self.preprocessor.process(title))
                doc_tokens.update(self.preprocessor.process(body))
                
            intersection = query_set.intersection(doc_tokens)
            union = query_set.union(doc_tokens)
            score = len(intersection) / len(union) if union else 0
            scores[doc_id] = score
            
        sorted_scores = sorted([(v, k) for k, v in scores.items()], key=lambda x: x[0], reverse=True)
        return sorted_scores[:k], {}
