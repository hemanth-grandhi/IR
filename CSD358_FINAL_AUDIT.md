# CSD358 Final Project Audit

## Assignment Compliance
PASS

## IR Components

| IR Concept | Status | File | Function | Notes |
| ---------- | ------ | ---- | -------- | ----- |
| Bag of Words | PASS | `src/index.py` | `InvertedIndex.add_document` | Uses a dictionary of postings to map terms to their occurrences. |
| Jaccard | PASS | `src/jaccard.py` | `JaccardBaseline.get_top_k` | Utilized as a baseline comparison in the evaluation script. |
| Log TF | PASS | `src/tfidf.py` | `TFIDFScorer._compute_tf` | `1 + log10(tf)` mathematically correct. |
| DF | PASS | `src/index.py` | `InvertedIndex.get_document_frequency` | Document Frequency correctly counts the number of docs with the term. |
| IDF | PASS | `src/tfidf.py` | `TFIDFScorer._compute_idf` | `log10(N / df)` mathematically correct. |
| TF-IDF | PASS | `src/tfidf.py` | `TFIDFScorer.build_vectors` | Combines term frequency and inverse document frequency properly. |
| Document vectors | PASS | `src/tfidf.py` | `TFIDFScorer.build_vectors` | Documents are represented dynamically in vector space. |
| Query vectors | PASS | `src/tfidf.py` | `TFIDFScorer.get_query_vector` | Queries are converted using compatible log TF and IDF spaces. |
| Length normalization | PASS | `src/tfidf.py` | `TFIDFScorer.build_vectors` | L2 normalization (`sqrt(sum(w_i^2))`) applied for cosine sim. |
| Cosine similarity | PASS | `src/ranking.py` | `Ranker.score_document` | Accurately calculates dot product divided by norm products. |
| SMART lnc.ltc | PASS | `src/tfidf.py` | `TFIDFScorer.build_vectors` | Available via `--smart`. Document uses log-TF (no IDF), query uses log-TF + IDF. |
| Ranking | PASS | `src/ranking.py` | `Ranker.get_top_k` | Assigns scores accurately via cosine and handles title/body zone weighting. |
| Top-K | PASS | `src/ranking.py` | `Ranker.get_top_k` | Leverages efficient `heapq.nlargest` for O(N log K) selection. |
| Evaluation | PASS | `src/evaluation.py` | `evaluate_system` | Generates queries, computes P@5, Recall, F1 against baselines. |

## Selected Track Compliance
PASS - The system is an academic literature search engine parsing document zones (title vs abstract/body) representing Track 6 (Vertical Search).

## Working System
PASS - System operates end-to-end, fetches live arXiv data, performs indexing, ranking, and outputs explainable insights.

## Evaluation
PASS - Comprehensive comparative evaluation exists testing Boolean AND, Boolean OR, Jaccard, Standard TF-IDF + Cosine, and SMART lnc.ltc.

## README
PASS - Properly documents the system, problem, concepts, and provides exact commands.

## Demo Readiness
PASS - `DEMO_GUIDE.md` added. Debug flags expose the necessary variables to satisfy assignment guidelines.

## Remaining TODOs
None.

## Critical Issues
None.

## Recommended Improvements
*   P2 (Optional): Swap out pseudo-qrels for a manually labeled subset of queries (e.g. 5-10 human-judged test queries) to show more accurate baseline comparisons.
*   P2 (Optional): Handle synonyms for stronger academic context (e.g., matching "ML" with "Machine Learning").
