import streamlit as st
import requests
import pandas as pd
from utils import MovieRecommender
from llm_utils import GeminiHandler

# --- Config ---
st.set_page_config(page_title="CineMatch Live", page_icon="🎬", layout="wide")
st.markdown("""
    <style>
    .movie-card { background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #FF4B4B; }
    .title { font-size: 1.2rem; font-weight: bold; }
    .meta { color: #aaa; font-size: 0.9rem; }
    .rating { font-weight: bold; color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

# --- Helper: TMDB Poster ---
def get_poster_url(poster_path):
    """
    Constructs the full image URL from the path saved in our dataframe.
    """
    if not poster_path or str(poster_path) == 'nan' or poster_path == '':
        return "https://via.placeholder.com/150x225?text=No+Poster"
    
    # TMDB base URL for images (w154 is a small, efficient size)
    return f"https://image.tmdb.org/t/p/w154{poster_path}"

@st.cache_resource
def load_models():
    return MovieRecommender(), GeminiHandler()

try:
    engine, ai = load_models()
except:
    st.error("⚠️ Run 'python build_tmdb_dataset.py' first!")
    st.stop()

# --- UI ---
st.title("CineMatch")
st.caption("Powered by Live TMDB Data + Semantic Search + Gemini AI")

query = st.text_input("What are you in the mood for?", placeholder="e.g. 'A dark psychological thriller released recently'")

if st.button("Recommend", type="primary"):
    if not query:
        st.warning("Type something!")
    else:
        with st.spinner("Searching fresh database..."):
            results = engine.hybrid_recommend(query)
            ai_summary = ai.generate_summary(query, results)
        
        if ai_summary:
            st.info(f"🤖 **Gemini Insight:** {ai_summary}")
            
        for _, row in results.iterrows():
            c1, c2 = st.columns([1, 4])
            with c1:
                st.image(get_poster_url(row['poster_path']))
            with c2:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="title">{row['title']} ({str(row['release_date'])[:4]})</div>
                    <div class="meta">{row['overview'][:150]}...</div>
                    <br>
                    <span class="rating">⭐ {row['rating']}/10</span> | Match Score: {row['score']:.2f}
                </div>
                """, unsafe_allow_html=True)