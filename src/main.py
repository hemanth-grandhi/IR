import argparse
import os
import sys

from preprocessing import Preprocessor
from index import InvertedIndex
from tfidf import TFIDFScorer
from ranking import Ranker
from explanation import Explainer

def print_debug_info(idx, scorer, preprocessor, query, query_tokens, query_vec, top_k):
    print("\n" + "="*40)
    print("IR DEBUG MODE")
    print("="*40)
    print(f"Vocabulary Size: {len(idx.index)}")
    print(f"Total Documents: {idx.total_docs}")
    print(f"Original Query: '{query}'")
    print(f"Tokens: {query_tokens}")
    print("-" * 40)
    
    for token in query_tokens:
        df = idx.get_document_frequency(token)
        idf = scorer.idf_map.get(token, 0)
        weight = query_vec.get(token, 0)
        print(f"Term: '{token}'")
        print(f"  DF: {df}")
        print(f"  IDF: {idf:.4f}")
        print(f"  Query Weight: {weight:.4f}")
        idx.print_debug_postings(token)
        print("-" * 40)
        
    print("Top retrieved documents and their scores:")
    for score, doc_id in top_k[:5]:
        print(f"  {doc_id} -> {score:.4f}")
    print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Academic Search Engine (CSD358 IR Hackathon)")
    parser.add_argument("--data", type=str, default="../data/arxiv_corpus.json", help="Path to corpus JSON")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--debug", action="store_true", help="Enable IR DEBUG mode to show intermediate outputs")
    parser.add_argument("--smart", action="store_true", help="Use SMART lnc.ltc weighting instead of standard TF-IDF")
    
    args = parser.parse_args()
    
    corpus_path = os.path.join(os.path.dirname(__file__), args.data)
    if not os.path.exists(corpus_path):
        print(f"Error: Dataset not found at {corpus_path}")
        print("Please run fetch_dataset.py first.")
        sys.exit(1)
        
    print("Initializing Information Retrieval Pipeline...")
    # 1. Preprocessor
    preprocessor = Preprocessor()
    
    # 2. Inverted Index
    idx = InvertedIndex(preprocessor)
    idx.build_from_corpus(corpus_path)
    
    # 3. TF-IDF Scorer
    weighting_scheme = "lnc.ltc" if args.smart else "ltc.ltc"
    scorer = TFIDFScorer(idx, smart_weighting=weighting_scheme)
    scorer.build_vectors()
    
    # 4. Ranker (with zone weighting)
    ranker = Ranker(scorer, title_weight=0.7, body_weight=0.3)
    
    print(f"\nSystem ready! (Weighting: {weighting_scheme}) Type your query or 'quit' to exit.")
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
                
            # Search pipeline
            query_tokens = preprocessor.process(query)
            if not query_tokens:
                print("Query contained only stop words or punctuation. Please try again.")
                continue
                
            top_k, contributions = ranker.get_top_k(query_tokens, k=args.k)
            
            # Debug Mode
            if args.debug:
                query_vec, _ = scorer.get_query_vector(query_tokens)
                print_debug_info(idx, scorer, preprocessor, query, query_tokens, query_vec, top_k)
                
            # Result Assembly & Explanation
            print(f"\nFound {len(top_k)} results for '{query}':")
            print("-" * 50)
            
            for rank, (score, doc_id) in enumerate(top_k, 1):
                metadata = idx.doc_metadata[doc_id]
                explanation = Explainer.format_explanation(doc_id, metadata, score, contributions[doc_id])
                print(f"Rank {rank}:")
                print(explanation)
                print("-" * 50)
                
        except KeyboardInterrupt:
            break
            
if __name__ == "__main__":
    main()
