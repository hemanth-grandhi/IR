# CSD358 IR Hackathon: Explainable Academic Search

## 1. Project Title

**Explainable Academic Search using Inverted Index, TF-IDF, Cosine Similarity, and Zone Weighting**

## 2. Problem Statement

Researchers often struggle to understand *why* a particular academic paper was recommended to them by a search engine. Traditional search systems may return relevant documents without clearly showing how the ranking was produced.

Our project addresses this problem by building an explainable academic search engine that retrieves relevant research papers and shows the contribution of matched query terms to the final ranking score.

## 3. Motivation

We wanted to build an interpretable search engine that not only retrieves relevant academic papers but also explains **why a document received its ranking**.

The system analyzes query terms and shows where they occur in the document, particularly in the **title and abstract/body zones**. This makes the retrieval process easier to understand compared with a system that only returns a ranked list of papers.

## 4. Track Relevance

This project primarily fits **T6: Vertical Search for Science**, as it searches academic papers and uses document structure such as titles and abstracts through zone weighting.

The project also focuses on trustworthy and explainable retrieval by showing why documents receive their ranking scores.

## 5. IR Concepts Used

The project implements the following Information Retrieval concepts:

1. **Case Folding & Tokenization** — `src/preprocessing.py`
2. **Stop-word Removal & Stemming** — `src/preprocessing.py`
3. **Inverted Index with Postings** — `src/index.py`
4. **Document Frequency (DF) and Inverse Document Frequency (IDF)** — `src/tfidf.py`
5. **Log-Frequency Term Frequency (TF)** — `src/tfidf.py`
6. **TF-IDF Weighting** — `src/tfidf.py`
7. **Vector Space Model** — `src/tfidf.py` and `src/ranking.py`
8. **Cosine Similarity** — `src/ranking.py`
9. **L2 Vector Normalization** — `src/tfidf.py`
10. **Zone Weighting for Title and Abstract/Body** — `src/ranking.py`
11. **Boolean AND/OR Retrieval** — baseline retrieval
12. **Jaccard Similarity** — `src/jaccard.py`
13. **SMART `lnc.ltc` Weighting** — `src/tfidf.py`
14. **Top-K Retrieval using a Heap** — `src/ranking.py`
15. **Score Explanation and Term-Level Contributions** — `src/explanation.py`

## 6. Architecture / Pipeline

The system follows the following retrieval pipeline:

1. `fetch_dataset.py` queries the arXiv API and builds a JSON academic-paper corpus.
2. `preprocessing.py` performs tokenization, case folding, stop-word removal, and stemming.
3. `index.py` builds the inverted index and stores term information in document postings.
4. `tfidf.py` calculates term frequency, document frequency, IDF, TF-IDF weights, SMART weighting, and document vector normalization.
5. `ranking.py` performs cosine similarity between query and document vectors, applies zone weighting, and retrieves the Top-K documents efficiently.
6. `explanation.py` decomposes the ranking score and shows the contribution of matched query terms.
7. `evaluation.py` evaluates the retrieval systems against the generated relevance sets.
8. `main.py` provides the command-line interface and debugging mode.
9. `app.py` provides the Streamlit web interface for interactive searching.

### Overall Pipeline

```text
Academic Papers
      |
      v
Dataset Collection
(fetch_dataset.py)
      |
      v
Preprocessing
(Tokenization, Case Folding,
 Stop-word Removal, Stemming)
      |
      v
Inverted Index
(index.py)
      |
      v
TF-IDF / SMART Weighting
(tfidf.py)
      |
      v
Vector Representation
      |
      v
Cosine Similarity
+ Zone Weighting
(ranking.py)
      |
      v
Top-K Results
      |
      v
Score Explanation
(explanation.py)
      |
      v
Explainable Academic Search
```

## 7. Dataset

* **Source:** arXiv API
* **Size:** 300 academic papers
* **Domain:** Computer Science research papers, including areas such as machine learning, information retrieval, distributed computing, and related fields
* **Fields Used:** `document_id`, `title`, `body` (abstract), `authors`, `year`

The dataset is stored locally as a JSON corpus and is used by the retrieval and evaluation components.

## 8. Installation

### Requirements

* Python 3.8 or higher
* pip

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If the repository is being cloned from GitHub:

```bash
git clone https://github.com/hemanth-grandhi/IR.git
cd IR
pip install -r requirements.txt
```

## 9. How to Run

### 1. Fetch the Dataset

A sample dataset is already provided in the repository. A fresh dataset can also be generated using:

```bash
python src/fetch_dataset.py
```

### 2. Run the Interactive CLI

```bash
python src/main.py
```

### 3. Run with IR Debug Mode

The debug mode displays intermediate retrieval information such as postings, weights, and ranking details:

```bash
python src/main.py --debug
```

For SMART weighting:

```bash
python src/main.py --smart --debug
```

### 4. Run the Streamlit Web Interface

```bash
streamlit run src/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

### 5. Run Evaluation

```bash
python src/evaluation.py
```

The evaluation results are saved in the `results/` directory.

## 10. Example Queries

The system can process academic search queries such as:

* `information retrieval`
* `distributed training for neural networks`
* `machine learning for image classification`
* `deep learning for image classification`
* `transformer models for natural language processing`
* `database index architecture`
* `computer vision`
* `federated learning`

## 11. Example Output

An example of the explainable retrieval output is:

```text
Found 5 results for 'distributed training':
--------------------------------------------------
Rank 1:
Document: Distributed Training of Deep Neural Networks
Score: 0.8421

Why this ranked highly:
  • 'distribut' → high contribution
    Matched in: title, body

  • 'train' → medium contribution
    Matched in: body
--------------------------------------------------
```

The actual scores and documents depend on the dataset and query being processed.

The explanation component connects the ranking score with individual query terms and their document zones, making the retrieval process more interpretable.

## 12. Evaluation

We evaluated multiple retrieval approaches using a pseudo-ground-truth relevance dataset generated from the academic corpus.

The evaluation compares simple retrieval baselines with vector-space retrieval approaches:

* Boolean AND
* Boolean OR
* Jaccard Similarity
* TF-IDF + Cosine Similarity
* TF-IDF + Zone Weighting
* SMART `lnc.ltc` + Zone Weighting

### Evaluation Results

| System                  | Precision@5 |     Recall | Overall Precision |   F1 Score |
| ----------------------- | ----------: | ---------: | ----------------: | ---------: |
| Boolean AND Baseline    |      0.4857 |     0.5140 |            0.6036 |     0.4948 |
| Boolean OR Baseline     |      0.2286 |     0.2901 |            0.2286 |     0.1792 |
| Jaccard Baseline        |      0.4571 |     0.5687 |            0.4571 |     0.4068 |
| **TF-IDF + Cosine**     |  **0.5000** | **0.6416** |            0.5000 | **0.4347** |
| TF-IDF + Zone Weighting |      0.4429 |     0.5416 |            0.4429 |     0.3653 |
| SMART `lnc.ltc` + Zone  |      0.4143 |     0.4767 |            0.4143 |     0.3197 |

### Evaluation Interpretation

The **TF-IDF + Cosine Similarity** system achieved the strongest overall retrieval performance among the evaluated systems, with:

* **Precision@5:** 0.5000
* **Recall:** 0.6416
* **F1 Score:** 0.4347

Compared with the simpler Jaccard and Boolean approaches, the vector-space model considers both **term importance and vector similarity**, allowing documents to be ranked according to their overall similarity to the query.

The results also show that adding zone weighting did not improve the measured performance on this particular evaluation dataset. This is an important observation because it demonstrates that an additional IR technique does not automatically improve retrieval performance for every dataset.

### Evaluation Limitation

The relevance sets were generated using heuristics based on query terms appearing in document titles and abstracts. Therefore, they represent **pseudo-ground truth rather than manually judged relevance**.

A future evaluation can use human relevance judgments from multiple reviewers to produce a stronger evaluation dataset.

## 13. Novelty

### Explainable IR Ranking

The main novelty of the project is **explainable retrieval**.

Instead of only returning a document and its similarity score, the system decomposes the ranking calculation and attributes portions of the score to individual query terms.

For example, a document may receive a high score because terms such as:

```text
"deep"
"learning"
"classification"
```

match strongly with the document.

The system also identifies whether the terms were matched in the **title or abstract/body**, allowing users to understand which parts of the academic document contributed to its ranking.

The system supports multiple weighting approaches, including standard TF-IDF/`ltc.ltc` and SMART `lnc.ltc` weighting.

## 14. Limitations

The current prototype has several limitations:

* It does not handle synonyms or semantic equivalence, such as `"ML"` and `"Machine Learning"`.
* The evaluation relevance sets are heuristically generated rather than manually judged by domain experts.
* The current corpus contains only 300 academic papers.
* The inverted index is held in memory, so it is not designed for very large-scale production search.
* Zone weighting did not improve the evaluation score on the current dataset.
* The system primarily performs lexical retrieval and therefore may miss documents that are semantically relevant but use different terminology.

## 15. Future Work

- Implement **BM25** scoring for improved ranking and document-length normalization.
- Add **query expansion** using techniques such as WordNet or pseudo-relevance feedback.
- Increase the size and diversity of the academic corpus.
- Introduce semantic retrieval to handle synonyms and related concepts.
- Improve the Streamlit web interface with additional filtering, richer explanations, and interactive evaluation visualization.
- Create a manually judged relevance dataset using human evaluators.
- Explore scalable on-disk indexing for larger academic collections.
## 16. Team Work Division

* **Member 1:** Preprocessing & Inverted Index — `preprocessing.py`, `index.py`
* **Member 2:** TF-IDF & Ranking Engine — `tfidf.py`, `ranking.py`
* **Member 3:** Explainability & Evaluation — `explanation.py`, `evaluation.py`
* **Member 4:** Dataset Integration & CLI — `fetch_dataset.py`, `main.py`

## 17. AI-Use Declaration

AI coding assistants were used during the development of this prototype for assistance with code development, debugging, documentation, and project organization.

The project team reviewed, tested, and integrated the generated suggestions into the final implementation.

Further details about AI usage are provided in:

```text
AI_USE.md
```

## 18. Repository Structure

```text
IR/
│
├── data/
│   └── arxiv_corpus.json
│
├── docs/
│
├── results/
│   ├── evaluation_results.md
│   └── p_at_5_comparison.png
│
├── src/
│   ├── app.py
│   ├── evaluation.py
│   ├── explanation.py
│   ├── fetch_dataset.py
│   ├── index.py
│   ├── jaccard.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── ranking.py
│   └── tfidf.py
│
├── tests/
│   └── test_pipeline.py
│
├── .gitignore
├── AI_USE.md
├── CSD358_FINAL_AUDIT.md
├── DEMO_GUIDE.md
├── README.md
├── RUBRIC_CHECKLIST.md
└── requirements.txt
```

## 19. Conclusion

This project demonstrates a complete academic information retrieval pipeline, beginning with document preprocessing and inverted-index construction and continuing through TF-IDF weighting, vector-space ranking, cosine similarity, zone weighting, Top-K retrieval, and explainable score generation.

The evaluation shows that **TF-IDF + Cosine Similarity** achieved the strongest performance among the tested systems. At the same time, the explainability component provides users with a clearer understanding of why individual academic papers are ranked highly.

The project therefore combines classical Information Retrieval techniques with an explainability layer to create a more transparent academic search experience.
