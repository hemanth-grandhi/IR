import streamlit as st
import os

from preprocessing import Preprocessor
from index import InvertedIndex
from tfidf import TFIDFScorer
from ranking import Ranker
from explanation import Explainer

# Cache the data loading and indexing so it doesn't reload on every interaction
@st.cache_resource
def load_ir_system():
    corpus_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'arxiv_corpus.json')
    
    preprocessor = Preprocessor()
    idx = InvertedIndex(preprocessor)
    idx.build_from_corpus(corpus_path)
    
    return preprocessor, idx

st.set_page_config(page_title="Academic Search Engine", layout="wide")

st.title("Search Explainable Academic Engine")
st.markdown("""
This is the CSD358 IR Hackathon demo. It uses an Inverted Index, TF-IDF, Cosine Similarity, and Zone Weighting to find the most relevant academic papers. 
It also explains **why** a paper was retrieved.
""")

with st.spinner("Initializing IR System (Loading Index & Vectors)..."):
    preprocessor, idx = load_ir_system()

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of results (Top-K)", min_value=1, max_value=20, value=5)
enable_debug = st.sidebar.checkbox("Enable IR Debug Mode", value=False)
enable_smart = st.sidebar.checkbox("Use SMART lnc.ltc Weighting", value=True)

# Build scorer dynamically based on settings
weighting_mode = "lnc.ltc" if enable_smart else "ltc.ltc"
scorer = TFIDFScorer(idx, smart_weighting=weighting_mode)
scorer.build_vectors()
ranker = Ranker(scorer, title_weight=0.7, body_weight=0.3)

query = st.text_input("Enter your research query:", placeholder="e.g., distributed training for neural networks")

if query:
    query_tokens = preprocessor.process(query)
    
    if not query_tokens:
        st.warning("Query contained only stop words or punctuation. Please try again.")
    else:
        results, contributions = ranker.get_top_k(query_tokens, k=top_k)
        
        st.subheader(f"Found {len(results)} results for '{query}'")
        
        # Display Results
        for rank, (score, doc_id) in enumerate(results, 1):
            metadata = idx.doc_metadata[doc_id]
            title = metadata.get('title', 'Unknown Title')
            authors = ", ".join(metadata.get('authors', []))
            year = metadata.get('year', '')
            body = metadata.get('body', '')
            
            with st.expander(f"**#{rank}** - {title} (Score: {score:.4f})", expanded=(rank==1)):
                st.caption(f"**Authors:** {authors} | **Year:** {year} | **ID:** {doc_id}")
                st.markdown("**Abstract:**")
                st.write(body)
                
                st.markdown("---")
                st.markdown("**Explainability - Why this ranked highly:**")
                
                sorted_terms = sorted(contributions[doc_id].items(), key=lambda x: x[1]['total'], reverse=True)
                for term, scores in sorted_terms[:5]:
                    total = scores['total']
                    if total > 0:
                        impact = "High" if total > 0.3 else "Medium" if total > 0.1 else "Low"
                        zones = []
                        if scores['title_score'] > 0: zones.append("Title")
                        if scores['body_score'] > 0: zones.append("Body")
                        
                        st.markdown(f"- **`{term}`** -> {impact} contribution ({total:.4f}) *Matched in: {', '.join(zones)}*")

        # Debug Mode Display
        if enable_debug:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Debug IR Information")
            st.sidebar.text(f"Vocabulary Size: {len(idx.index)}")
            st.sidebar.text(f"Total Documents: {idx.total_docs}")
            st.sidebar.text(f"Processed Tokens: {query_tokens}")
            
            query_vec, _ = scorer.get_query_vector(query_tokens)
            
            st.sidebar.markdown("**Term Dictionary Stats:**")
            for token in query_tokens:
                df = idx.get_document_frequency(token)
                idf = scorer.idf_map.get(token, 0)
                st.sidebar.markdown(f"- `{token}`: DF={df}, IDF={idf:.4f}")
                
            st.sidebar.markdown("**Query Vector (TF-IDF):**")
            st.sidebar.json(query_vec)
