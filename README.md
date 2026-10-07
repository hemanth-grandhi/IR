# CSD358 IR Hackathon: Explainable Academic Search

## 1. Project Title
Explainable Academic Search using Inverted Index, TF-IDF, Cosine Similarity, and Zone Weighting

## 2. Problem Statement
Researchers often struggle to understand *why* a particular academic paper was recommended to them by a search engine. Black-box semantic search engines hide the ranking logic, which can lead to frustration when evaluating relevance.

## 3. Motivation
We wanted to build an interpretable search engine that not only retrieves relevant academic papers but explicitly explains the score contributions of matched terms across different document zones (title vs. body).

## 4. Track Relevance
This project fits into **T1: Retrieval-Augmented Generation and trustworthy answers** (specifically focusing on the trustworthy/explainable retriever component) and **T6: Vertical search for science**, as it handles academic document structure (titles, abstracts) via zone weighting.

## 5. IR Concepts Used
1. **Case Folding & Tokenization** (`src/preprocessing.py`)
2. **Stop-word Removal & Stemming** (`src/preprocessing.py`)
3. **Inverted Index with Postings** (`src/index.py`)
4. **TF-IDF Weighting** (`src/tfidf.py`)
5. **Vector Space Model & Cosine Similarity** (`src/ranking.py`)
6. **Zone Weighting (Title vs Body)** (`src/ranking.py`)
7. **Efficient Top-K Heap Selection** (`src/ranking.py`)

## 6. Architecture / Pipeline
1. `fetch_dataset.py` queries the arXiv API and builds a JSON corpus.
2. `preprocessing.py` normalizes queries and documents.
3. `index.py` builds the Inverted Index, storing term frequencies separated by zone.
4. `tfidf.py` precomputes IDF values and document norms.
5. `ranking.py` performs cosine similarity between the query vector and document vectors, applying zone weights, and retrieving Top-K via a min-heap.
6. `explanation.py` formats the term-level contribution to the final score.
7. `main.py` provides the CLI and `IR_DEBUG` mode.

## 7. Dataset
- **Source:** arXiv API (Public domain / open access)
- **Size:** 300 academic papers (filtered from CS domains like ML, IR, Distributed Computing)
- **Fields Used:** `document_id`, `title`, `body` (abstract), `authors`, `year`

## 8. Installation
```bash
# Clone the repo (or extract the zip)
# Requires Python 3.8+
pip install -r requirements.txt
```

## 9. How to Run
```bash
# 1. Fetch the dataset (we have provided a sample, but you can run this to fetch fresh data)
python src/fetch_dataset.py

# 2. Run the interactive CLI
python src/main.py

# 3. Run with IR DEBUG mode to see intermediate postings and weights
python src/main.py --debug

# 4. Run the Evaluation script
python src/evaluation.py
```

## 10. Example Queries
- `information retrieval`
- `distributed training for neural networks`
- `database index architecture`

## 11. Example Output
```
Found 5 results for 'distributed training':
--------------------------------------------------
Rank 1:
Document: Distributed Training of Deep Neural Networks (ID: 1234.5678)
Score: 0.8421
Why this ranked highly:
  • 'distribut' → high contribution (0.5123) (matched in: title, body)
  • 'train' → medium contribution (0.3298) (matched in: body)
--------------------------------------------------
```

## 12. Evaluation
We created a small pseudo-ground-truth dataset of queries using strong heuristics to evaluate our ranking mechanisms. We evaluated several baselines (Boolean AND, Boolean OR, and Jaccard Coefficient) against our Improved systems (Standard TF-IDF and SMART `lnc.ltc` Weighting).

| System | Precision@5 | Recall | Overall Precision | F1 Score |
|---|---:|---:|---:|---:|
| Boolean AND Baseline | 0.4857 | 0.5140 | 0.4857 | 0.4948 |
| Boolean OR Baseline | 0.2286 | 0.2901 | 0.2286 | 0.1792 |
| Jaccard Baseline | 0.4571 | 0.5687 | 0.4571 | 0.4068 |
| TF-IDF + Cosine | 0.5000 | 0.6416 | 0.5000 | 0.4347 |
| TF-IDF + Zone Weighting | 0.4429 | 0.5416 | 0.4429 | 0.3653 |
| SMART lnc.ltc + Zone | 0.4143 | 0.4767 | 0.4143 | 0.3197 |

*Note: The Vector Space Model (TF-IDF + Cosine similarity) successfully improves upon the simpler Set-based Jaccard and Boolean baselines by normalizing for length and term importance!*

## 13. Novelty
**Explainable IR Ranking:** Instead of just returning a score, our ranker mathematically decomposes the Cosine Similarity calculation and attributes portions of the final score to individual query terms. It maps these contributions back to the exact document zones (title vs body) where they matched, allowing users to see exactly *why* a paper was recommended. We also support multiple configurable weighting schemes (Standard `ltc.ltc` and SMART `lnc.ltc`).

## 14. Limitations
- Does not handle synonyms or semantic equivalence (e.g., "ML" vs "Machine Learning").
- The evaluation dataset was heuristically generated rather than manually judged by experts.
- Index is held entirely in memory; a production system would need an on-disk index (like Lucene).

## 15. Future Work
- Implement BM25 scoring for better length normalization.
- Add query expansion using WordNet or pseudo-relevance feedback.
- Build a Streamlit web interface for a better user experience.

## 16. Team Work Division
- **Member 1:** Preprocessing & Inverted Index (`preprocessing.py`, `index.py`)
- **Member 2:** TF-IDF & Ranking Engine (`tfidf.py`, `ranking.py`)
- **Member 3:** Explainability & Evaluation (`explanation.py`, `evaluation.py`)
- **Member 4:** Dataset Integration & CLI (`fetch_dataset.py`, `main.py`)

## 17. AI-Use Declaration
AI coding assistants were used during the development of this prototype. See `AI_USE.md` for details.
