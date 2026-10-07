import json
import os
import matplotlib.pyplot as plt

def load_queries_and_qrels(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
        
    queries = [
        "distributed training for neural networks",
        "machine learning for image classification",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "neural network optimization",
        "deep learning",
        "graph neural networks",
        "transformer models",
        "medical image analysis",
        "recommendation systems",
        "federated learning",
        "anomaly detection",
        "information retrieval",
        "large language models",
        "database index architecture",
        "gpu acceleration",
        "semantic search",
        "data mining"
    ]
    
    qrels = {}
    for q in queries:
        q_terms = q.lower().split()
        relevant_docs = set()
        
        for doc in docs:
            title_lower = doc['title'].lower()
            body_lower = doc['body'].lower()
            
            if q.lower() in title_lower or q.lower() in body_lower:
                relevant_docs.add(doc['document_id'])
            elif all(term in title_lower for term in q_terms):
                relevant_docs.add(doc['document_id'])
                
        qrels[q] = relevant_docs
        
    valid_qrels = {q: rels for q, rels in qrels.items() if len(rels) > 0}
    return valid_qrels

def evaluate_system(ranker, preprocessor, qrels, k=5):
    metrics = {'p_at_k': 0.0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
    
    for q, relevant_docs in qrels.items():
        tokens = preprocessor.process(q)
        top_k, _ = ranker.get_top_k(tokens, k=k)
        
        retrieved_docs = [doc_id for score, doc_id in top_k]
        
        hits = sum(1 for d in retrieved_docs if d in relevant_docs)
        
        p_at_k = hits / k if k > 0 else 0
        recall = hits / len(relevant_docs) if len(relevant_docs) > 0 else 0
        precision = hits / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['p_at_k'] += p_at_k
        metrics['recall'] += recall
        metrics['precision'] += precision
        metrics['f1'] += f1
        
    num_q = len(qrels)
    if num_q > 0:
        metrics['p_at_k'] /= num_q
        metrics['recall'] /= num_q
        metrics['precision'] /= num_q
        metrics['f1'] /= num_q
        
    return metrics
    
class BooleanAndBaseline:
    """A Boolean AND retrieval baseline (postings intersection)."""
    def __init__(self, inverted_index, preprocessor):
        self.index = inverted_index
        self.preprocessor = preprocessor
        
    def get_top_k(self, query_tokens, k=10):
        if not query_tokens:
            return [], {}
            
        tokens_by_df = sorted(query_tokens, key=lambda t: len(self.index.get_postings(t)))
        
        first_postings = self.index.get_postings(tokens_by_df[0])
        intersected_docs = set(first_postings.keys())
        
        for token in tokens_by_df[1:]:
            postings = self.index.get_postings(token)
            intersected_docs = intersected_docs.intersection(set(postings.keys()))
            if not intersected_docs:
                break
                
        scores = {}
        for doc_id in intersected_docs:
            score = 0
            for token in query_tokens:
                postings = self.index.get_postings(token)
                score += postings[doc_id]['title_tf'] + postings[doc_id]['body_tf']
            scores[doc_id] = score
            
        sorted_scores = sorted([(v, k) for k, v in scores.items()], key=lambda x: x[0], reverse=True)
        return sorted_scores[:k], {}

class BooleanOrBaseline:
    """A Boolean OR retrieval baseline (any matching term)."""
    def __init__(self, inverted_index, preprocessor):
        self.index = inverted_index
        self.preprocessor = preprocessor
        
    def get_top_k(self, query_tokens, k=10):
        if not query_tokens:
            return [], {}
            
        scores = {}
        for token in query_tokens:
            postings = self.index.get_postings(token)
            for doc_id, tfs in postings.items():
                scores[doc_id] = scores.get(doc_id, 0) + tfs['title_tf'] + tfs['body_tf']
                
        sorted_scores = sorted([(v, k) for k, v in scores.items()], key=lambda x: x[0], reverse=True)
        return sorted_scores[:k], {}

from jaccard import JaccardBaseline

if __name__ == "__main__":
    from preprocessing import Preprocessor
    from index import InvertedIndex
    from tfidf import TFIDFScorer
    from ranking import Ranker
    
    corpus_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'arxiv_corpus.json')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading data and building index...")
    p = Preprocessor()
    idx = InvertedIndex(p)
    idx.build_from_corpus(corpus_file)
    
    # TFIDF Scorer (Standard ltc.ltc)
    scorer = TFIDFScorer(idx, smart_weighting="ltc.ltc")
    scorer.build_vectors()

    # SMART Scorer (lnc.ltc)
    smart_scorer = TFIDFScorer(idx, smart_weighting="lnc.ltc")
    smart_scorer.build_vectors()
    
    qrels = load_queries_and_qrels(corpus_file)
    print(f"\nGenerated {len(qrels)} viable queries for evaluation.")
    
    print("\n--- Evaluation Results ---")
    
    baseline_and = BooleanAndBaseline(idx, p)
    res_base_and = evaluate_system(baseline_and, p, qrels, k=5)
    print(f"Boolean AND Baseline     -> P@5: {res_base_and['p_at_k']:.4f}, Recall: {res_base_and['recall']:.4f}, F1: {res_base_and['f1']:.4f}")
    
    baseline_or = BooleanOrBaseline(idx, p)
    res_base_or = evaluate_system(baseline_or, p, qrels, k=5)
    print(f"Boolean OR Baseline      -> P@5: {res_base_or['p_at_k']:.4f}, Recall: {res_base_or['recall']:.4f}, F1: {res_base_or['f1']:.4f}")

    baseline_jaccard = JaccardBaseline(idx, p)
    res_jaccard = evaluate_system(baseline_jaccard, p, qrels, k=5)
    print(f"Jaccard Baseline         -> P@5: {res_jaccard['p_at_k']:.4f}, Recall: {res_jaccard['recall']:.4f}, F1: {res_jaccard['f1']:.4f}")
    
    ranker_no_zone = Ranker(scorer, title_weight=0.5, body_weight=0.5)
    res_no_zone = evaluate_system(ranker_no_zone, p, qrels, k=5)
    print(f"TF-IDF + Cosine          -> P@5: {res_no_zone['p_at_k']:.4f}, Recall: {res_no_zone['recall']:.4f}, F1: {res_no_zone['f1']:.4f}")
    
    ranker_zone = Ranker(scorer, title_weight=0.7, body_weight=0.3)
    res_zone = evaluate_system(ranker_zone, p, qrels, k=5)
    print(f"TF-IDF + Zone Weighting  -> P@5: {res_zone['p_at_k']:.4f}, Recall: {res_zone['recall']:.4f}, F1: {res_zone['f1']:.4f}")

    ranker_smart = Ranker(smart_scorer, title_weight=0.7, body_weight=0.3)
    res_smart = evaluate_system(ranker_smart, p, qrels, k=5)
    print(f"SMART lnc.ltc + Zone     -> P@5: {res_smart['p_at_k']:.4f}, Recall: {res_smart['recall']:.4f}, F1: {res_smart['f1']:.4f}")
    
    print("\nSaving results...")
    md_path = os.path.join(results_dir, 'evaluation_results.md')
    with open(md_path, 'w') as f:
        f.write("# IR Evaluation Results\n\n")
        f.write(f"Number of queries evaluated: {len(qrels)}\n\n")
        f.write("| System | Precision@5 | Recall | Overall Precision | F1 Score |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        f.write(f"| Boolean AND Baseline | {res_base_and['p_at_k']:.4f} | {res_base_and['recall']:.4f} | {res_base_and['precision']:.4f} | {res_base_and['f1']:.4f} |\n")
        f.write(f"| Boolean OR Baseline | {res_base_or['p_at_k']:.4f} | {res_base_or['recall']:.4f} | {res_base_or['precision']:.4f} | {res_base_or['f1']:.4f} |\n")
        f.write(f"| Jaccard Baseline | {res_jaccard['p_at_k']:.4f} | {res_jaccard['recall']:.4f} | {res_jaccard['precision']:.4f} | {res_jaccard['f1']:.4f} |\n")
        f.write(f"| TF-IDF + Cosine | {res_no_zone['p_at_k']:.4f} | {res_no_zone['recall']:.4f} | {res_no_zone['precision']:.4f} | {res_no_zone['f1']:.4f} |\n")
        f.write(f"| TF-IDF + Zone Weighting | {res_zone['p_at_k']:.4f} | {res_zone['recall']:.4f} | {res_zone['precision']:.4f} | {res_zone['f1']:.4f} |\n")
        f.write(f"| SMART lnc.ltc + Zone | {res_smart['p_at_k']:.4f} | {res_smart['recall']:.4f} | {res_smart['precision']:.4f} | {res_smart['f1']:.4f} |\n")
        
    print(f"Results saved to {md_path}")
    
    try:
        labels = ['Bool AND', 'Bool OR', 'Jaccard', 'TF-IDF', 'TF-IDF+Zone', 'SMART lnc.ltc']
        p5_vals = [res_base_and['p_at_k'], res_base_or['p_at_k'], res_jaccard['p_at_k'], res_no_zone['p_at_k'], res_zone['p_at_k'], res_smart['p_at_k']]
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, p5_vals, color=['darkred', 'red', 'orange', 'blue', 'green', 'purple'])
        plt.ylabel('Precision@5')
        plt.title('IR System Performance Comparison')
        plt.ylim(0, 1.0)
        plot_path = os.path.join(results_dir, 'p_at_5_comparison.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print("Could not generate plot, ensure matplotlib is installed.", e)
