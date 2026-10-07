# Grading Rubric Self-Audit

This checklist maps our project features directly to the CSD358 Hackathon Rubric.

## 1. Use of IR principles (30 marks)
- [x] **Boolean Retrieval & Postings Intersection:** Baseline implemented using Boolean AND by intersecting postings.
- [x] **Query Optimization:** Boolean baseline processes terms in order of increasing Document Frequency (DF).
- [x] **Case folding & Tokenization:** Implemented in `src/preprocessing.py`
- [x] **Stop-word removal & Stemming:** Implemented in `src/preprocessing.py` (uses NLTK PorterStemmer)
- [x] **Inverted Index (Dictionary + Postings):** Implemented in `src/index.py` (stores zone-specific TFs)
- [x] **TF-IDF Weighting:** Implemented in `src/tfidf.py` (log frequency weighting)
- [x] **Cosine Similarity & Vector Space Model:** Implemented in `src/ranking.py`
- [x] **Zone Weighting:** Implemented in `src/ranking.py` (Title vs Body weights)
- [x] **Top-K Result Assembly:** Implemented via Min-Heap (`heapq.nlargest`) in `src/ranking.py`

## 2. Novelty and creativity (10 marks)
- [x] **Explainable IR Ranking:** Instead of just returning a score, the system breaks down exactly which query terms contributed to the score, whether they matched in the title or the body, and what their mathematical impact was. Implemented in `src/explanation.py`.

## 3. Working system (20 marks)
- [x] **Runs Live:** The system successfully loads 300 arXiv papers and runs searches instantly via `src/main.py`.
- [x] **No Hardcoding:** Results and scores are calculated dynamically from the index.
- [x] **Reproducible:** `fetch_dataset.py` allows downloading a fresh real-world dataset.

## 4. Evaluation metrics (15 marks)
- [x] **Queries & Judgements:** Evaluated against 10 queries using a heuristic pseudo-relevance strategy.
- [x] **P/R/F1/P@k:** Computes Precision, Recall, F1 Score, and P@5.
- [x] **Baseline Comparison:** Compares a naive term frequency matching baseline against the improved TF-IDF/Zone system, proving the efficacy of the IR principles. Implemented in `src/evaluation.py`.

## 5. Relevance to the track (5 marks)
- [x] **Track T6 (Vertical Search):** Specifically designed for academic papers, utilizing domain structure (Titles vs Abstracts).
- [x] **Track T1 (Trustworthy Answers):** The explainability feature explicitly traces ranking decisions back to the source terms.

## 6. Report quality (10 marks)
- [x] README and Evaluation output provide the necessary data and structure for the 8-page PDF report.
- [x] AI-Use declaration included.

## 7. Video explanation (10 marks)
- [x] `DEMO_GUIDE.md` created to structure the 5-8 minute presentation, ensuring all required concepts and intermediate outputs (`--debug` mode) are shown.
