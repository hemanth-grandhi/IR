# Demo Video Guide

Follow this guide to structure your 5–8 minute unlisted YouTube/Drive presentation. Do not use slides. You must show the actual working system, and team members must explain the components they built.

## 1. Problem + Track (0:00 - 1:00)
*   **Track:** T6 (Vertical search for law, finance or science).
*   **Problem:** Explain that researchers struggle to understand *why* a particular academic paper was recommended. Black-box semantic search engines hide the ranking logic.
*   **Solution:** We built an interpretable academic search engine over an arXiv corpus that decomposes and explains the Cosine Similarity score across different document zones (title vs body).

## 2. Corpus/Data (1:00 - 1:30)
*   Show `data/arxiv_corpus.json`.
*   Explain how `src/fetch_dataset.py` pulls data from the arXiv API.
*   Mention that it includes `document_id`, `title`, and `body` (abstract).

## 3. Real Query & Preprocessing (1:30 - 2:30)
*   Run the interactive CLI: `python src/main.py --debug --smart`
*   Enter a query such as: `distributed training for neural networks`
*   Show the **IR DEBUG MODE** output.
*   Point out how the query was preprocessed (e.g., lowercase, punctuation removed, stopwords like 'for' removed, resulting in `distribut`, `train`, `neural`, `network`).

## 4. TF/DF/IDF & Document/Query Vectors (2:30 - 4:00)
*   STILL looking at the DEBUG output for the query.
*   Show the Document Frequency (DF) and Inverse Document Frequency (IDF) for a term (e.g., 'distribut').
*   Show the **Query Weight**. Explain how this is calculated (`ltc` from SMART weighting).
*   Show the intermediate postings (term frequency inside the `title` and `body`).
*   Show how the Document Vectors are constructed in `src/tfidf.py` using `lnc` weighting (log TF, no IDF, cosine norm).

## 5. Cosine/Similarity Scores & Ranked Top-K (4:00 - 5:30)
*   Scroll down to the generated results.
*   Show the Top-K ranking list.
*   Explain the output of `explanation.py` which shows the exact score contribution for each matched term, broken down by title vs body.
*   Highlight that `heapq.nlargest` is used to get the top results efficiently in `src/ranking.py`.

## 6. Evaluation Output & Comparison (5:30 - 6:30)
*   Run the evaluation script: `python src/evaluation.py`
*   Show the evaluation results (`evaluation_results.md` and `p_at_5_comparison.png`).
*   Compare the baselines (Boolean AND, Boolean OR, Jaccard Coefficient) with the TF-IDF approaches (Standard `ltc.ltc` vs SMART `lnc.ltc`).
*   Discuss why VSM / SMART weighting outperforms simple Set-based approaches.

## 7. Limitations & Conclusion (6:30 - 7:30)
*   **Limitation:** The index is held entirely in memory; a production system needs an on-disk inverted index.
*   **Limitation:** Our evaluation dataset uses pseudo-qrels (heuristically generated) rather than manual expert judgments.
*   **Future Work:** Mention BM25 scoring or on-disk indices.
*   **Sign Off:** Briefly restate the team member contributions (who owned which file).
