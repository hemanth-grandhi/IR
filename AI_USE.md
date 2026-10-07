# AI-Use Declaration

As permitted by the CSD358 IR Hackathon assignment instructions, this project utilized AI coding assistants during development. 

## Details of AI Usage

**Tools Used:** Gemini 3.1 Pro (via Antigravity Agent) / GitHub Copilot

**How AI was used:**
1. **Boilerplate & Infrastructure:** Generating the directory structure, `argparse` CLI setup, and standard class definitions.
2. **Implementation:** Assisting with the implementation of core IR formulas (like TF, IDF, and Cosine Similarity) based on standard mathematical definitions.
3. **Data Fetching:** Writing the arXiv API fetching script (`fetch_dataset.py`) to quickly obtain a realistic corpus while adhering to rate limits.
4. **Debugging & Testing:** Generating Python `unittest` test cases and fixing edge-case bugs (e.g., handling queries with only stop words).
5. **Documentation:** Helping to draft the formatting of the README, DEMO_GUIDE, and this declaration.

**Human Validation:**
The team thoroughly reviewed all generated code to ensure it adhered to the strict IR principles taught in the CSD358 lectures. We manually verified that the mathematics in `tfidf.py` and `ranking.py` were correct, and we designed the Explainable IR logic and Zone Weighting concepts ourselves. No existing projects were copied, and all core logic is original to this project.
