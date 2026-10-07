import math

class TFIDFScorer:
    def __init__(self, inverted_index, smart_weighting="ltc.ltc"):
        self.index = inverted_index
        self.N = inverted_index.total_docs
        self.idf_map = {}
        self.doc_vectors = {} # {doc_id: {'title_vec': {term: weight}, 'body_vec': {term: weight}, 'title_norm': float, 'body_norm': float}}
        self.smart_weighting = smart_weighting
        
    def _compute_tf(self, raw_tf):
        """Log-frequency weighting: 1 + log10(tf)"""
        if raw_tf > 0:
            return 1 + math.log10(raw_tf)
        return 0
        
    def _compute_idf(self, df):
        """Inverse Document Frequency: log10(N / df)"""
        if df == 0:
            return 0
        return math.log10(self.N / df)
        
    def build_vectors(self):
        """Precomputes IDF and document vectors for faster cosine similarity."""
        print(f"Computing IDF and Document Vectors (Weighting: {self.smart_weighting})...")
        
        # 1. Compute IDF for all terms
        for term, postings in self.index.index.items():
            df = len(postings)
            self.idf_map[term] = self._compute_idf(df)
            
        # 2. Compute Document Vectors and their norms (for cosine normalization)
        for term, postings in self.index.index.items():
            idf = self.idf_map[term]
            for doc_id, tfs in postings.items():
                if doc_id not in self.doc_vectors:
                    self.doc_vectors[doc_id] = {
                        'title_vec': {}, 'body_vec': {}, 
                        'title_norm': 0.0, 'body_norm': 0.0
                    }
                    
                # For document (lnc.ltc -> doc is lnc -> tf only, ltc.ltc -> doc is ltc -> tf * idf)
                doc_idf = 1.0 if self.smart_weighting == "lnc.ltc" else idf

                # Title weight
                title_tf = self._compute_tf(tfs['title_tf'])
                if title_tf > 0:
                    w = title_tf * doc_idf
                    self.doc_vectors[doc_id]['title_vec'][term] = w
                    self.doc_vectors[doc_id]['title_norm'] += w * w
                    
                # Body weight
                body_tf = self._compute_tf(tfs['body_tf'])
                if body_tf > 0:
                    w = body_tf * doc_idf
                    self.doc_vectors[doc_id]['body_vec'][term] = w
                    self.doc_vectors[doc_id]['body_norm'] += w * w
                    
        # 3. Finalize norms (square root)
        for doc_id in self.doc_vectors:
            self.doc_vectors[doc_id]['title_norm'] = math.sqrt(self.doc_vectors[doc_id]['title_norm'])
            self.doc_vectors[doc_id]['body_norm'] = math.sqrt(self.doc_vectors[doc_id]['body_norm'])
            
        print("TF-IDF vectors built successfully.")
        
    def get_query_vector(self, query_tokens):
        """Builds TF-IDF vector for a query.
           For both lnc.ltc and ltc.ltc, the query is ltc (log TF * IDF).
        """
        # Calculate raw TF for query
        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
            
        query_vec = {}
        query_norm = 0.0
        
        for term, raw_tf in query_tf.items():
            tf = self._compute_tf(raw_tf)
            idf = self.idf_map.get(term, 0) # 0 if term not in corpus
            w = tf * idf
            if w > 0:
                query_vec[term] = w
                query_norm += w * w
                
        query_norm = math.sqrt(query_norm)
        return query_vec, query_norm
