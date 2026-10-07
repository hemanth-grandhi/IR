# IR Implementation Notes

## Information Retrieval Pipeline Mapping

This document provides a clear mapping of the Information Retrieval (IR) concepts implemented in this project to their corresponding exact file and function locations.

### 1. Tokenization and Normalization
*   **Purpose:** Standardizes the text representation by case folding, removing punctuation, splitting tokens by whitespace, and selectively removing stop words or applying stemming.
*   **File:** `src/preprocessing.py`
*   **Class/Function:** `Preprocessor.process(text)`
*   **Formula/Logic:** Lowers text, replaces punctuation with spaces, splits into tokens. Removes stop words based on a predefined set. Optionally stems using `PorterStemmer`.

### 2. Inverted Index (Bag of Words)
*   **Purpose:** Maps vocabulary terms to the documents that contain them, tracking occurrences for TF-IDF calculations. Implements a logical Bag of Words format over the corpus vocabulary.
*   **File:** `src/index.py`
*   **Class/Function:** `InvertedIndex.add_document` and `InvertedIndex.index`
*   **Formula/Logic:** `{ term: { doc_id: { 'title_tf': int, 'body_tf': int } } }`

### 3. Log-Frequency Term Frequency (TF)
*   **Purpose:** Dampens the effect of terms occurring many times in a single document (since relevance does not scale linearly with frequency).
*   **File:** `src/tfidf.py`
*   **Class/Function:** `TFIDFScorer._compute_tf(raw_tf)`
*   **Formula:** `1 + log10(tf)` if `tf > 0`, else `0`

### 4. Document Frequency (DF) & Inverse Document Frequency (IDF)
*   **Purpose:** Calculates term importance across the corpus. Rare terms have a high IDF, while common terms have a low IDF.
*   **File:** `src/index.py` and `src/tfidf.py`
*   **Class/Function:** `InvertedIndex.get_document_frequency(term)` and `TFIDFScorer._compute_idf(df)`
*   **Formula:** `log10(N / df)` where `N` is the total number of documents (`InvertedIndex.total_docs`).

### 5. Document & Query Vectors (Vector Space Model)
*   **Purpose:** Represents documents and queries as weighted vectors in a high-dimensional space where each dimension is a term from the vocabulary.
*   **File:** `src/tfidf.py`
*   **Class/Function:** `TFIDFScorer.build_vectors()` and `TFIDFScorer.get_query_vector(query_tokens)`

### 6. SMART lnc.ltc Weighting Scheme
*   **Purpose:** Provides a refined weighting model based on SMART notation.
    *   **Document (lnc):** logarithmic TF (`l`), no IDF (`n`), cosine normalization (`c`).
    *   **Query (ltc):** logarithmic TF (`l`), IDF weighting (`t`), cosine normalization (`c`).
*   **File:** `src/tfidf.py`
*   **Class/Function:** `TFIDFScorer.__init__` and `TFIDFScorer.build_vectors()`
*   **Formula/Logic:** Configured via `smart_weighting="lnc.ltc"`. The document uses `doc_idf = 1.0`, whereas standard `ltc.ltc` includes IDF for the document.

### 7. Length (L2) Normalization & Cosine Similarity
*   **Purpose:** Normalizes term weights by the L2 norm of the vector so that long documents are not unfairly advantaged. Used to compute Cosine Similarity as a dot product of normalized vectors.
*   **File:** `src/ranking.py`
*   **Class/Function:** `Ranker.score_document()`
*   **Formula:** `cosine_sim = dot_product / (query_norm * doc_norm)` where norm is `sqrt(sum(w_i^2))`.

### 8. Top-K Retrieval
*   **Purpose:** Retrieves only the highest-scoring `K` documents efficiently without fully sorting the entire corpus scores.
*   **File:** `src/ranking.py`
*   **Class/Function:** `Ranker.get_top_k()`
*   **Formula/Logic:** Implements a Min-Heap via Python's `heapq.nlargest()`, achieving `O(N log K)` complexity over candidate documents.

### 9. Jaccard Coefficient (Baseline)
*   **Purpose:** Used as a simpler evaluation baseline. Measures the size of the intersection divided by the size of the union of the query terms and document terms.
*   **File:** `src/jaccard.py`
*   **Class/Function:** `JaccardBaseline.get_top_k()`
*   **Formula:** `|Q ∩ D| / |Q ∪ D|`
