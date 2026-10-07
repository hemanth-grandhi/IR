from collections import defaultdict
import json
import os

class InvertedIndex:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        # Structure: { term: { doc_id: { 'title_tf': int, 'body_tf': int } } }
        self.index = defaultdict(lambda: defaultdict(lambda: {'title_tf': 0, 'body_tf': 0}))
        self.doc_lengths = {} # Stores document lengths (optional, for other scoring)
        self.doc_metadata = {} # Stores doc info like title, url, etc.
        self.total_docs = 0
        
    def add_document(self, doc_id, title, body, metadata=None):
        if doc_id in self.doc_metadata:
            return # Skip duplicates
            
        self.total_docs += 1
        
        # Save metadata
        if metadata is None:
            metadata = {}
        metadata['title'] = title
        metadata['body'] = body
        self.doc_metadata[doc_id] = metadata
        
        # Process title
        title_tokens = self.preprocessor.process(title)
        for token in title_tokens:
            self.index[token][doc_id]['title_tf'] += 1
            
        # Process body
        body_tokens = self.preprocessor.process(body)
        for token in body_tokens:
            self.index[token][doc_id]['body_tf'] += 1
            
    def get_postings(self, term):
        """Returns postings for a specific term."""
        token = self.preprocessor.process(term)
        if not token:
            return {}
        term = token[0] # assume single term
        return self.index.get(term, {})
        
    def get_document_frequency(self, term):
        """Returns DF (Document Frequency) for a term."""
        return len(self.get_postings(term))
        
    def build_from_corpus(self, corpus_path):
        """Builds index from a JSON list of documents."""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
            
        for doc in documents:
            self.add_document(
                doc['document_id'], 
                doc['title'], 
                doc['body'], 
                metadata={
                    'authors': doc.get('authors', []),
                    'year': doc.get('year', ''),
                    'source': doc.get('source', '')
                }
            )
        print(f"Index built! Total documents: {self.total_docs}, Vocabulary size: {len(self.index)}")

    def print_debug_postings(self, term):
        """Debug function as requested in requirements."""
        postings = self.get_postings(term)
        processed_term = self.preprocessor.process(term)[0] if self.preprocessor.process(term) else term
        print(f"\n--- Postings for '{processed_term}' ---")
        print(f"Document Frequency: {len(postings)}")
        for doc_id, tfs in list(postings.items())[:5]: # Show top 5 for debug
            print(f" Doc {doc_id}: title_tf={tfs['title_tf']}, body_tf={tfs['body_tf']}")
        if len(postings) > 5:
            print(f" ... and {len(postings) - 5} more.")

if __name__ == "__main__":
    from preprocessing import Preprocessor
    p = Preprocessor()
    idx = InvertedIndex(p)
    corpus_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'arxiv_corpus.json')
    if os.path.exists(corpus_file):
        idx.build_from_corpus(corpus_file)
        idx.print_debug_postings("distributed")
