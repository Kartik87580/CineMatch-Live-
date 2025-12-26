import requests
import pandas as pd
import numpy as np
import pickle
import faiss
import time
import os
import tomllib
from sentence_transformers import SentenceTransformer
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# --- CONFIG ---
SECRETS_PATH = ".streamlit/secrets.toml"
PAGES_TO_FETCH = 20  # Keep it moderate to avoid bans
OUTPUT_DIR = "models"

# --- GENRE MAPPING ---
GENRE_MAP = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}

# --- 1. Load API Key ---
def load_api_key():
    if not os.path.exists(SECRETS_PATH):
        print(f"❌ Error: Could not find secrets file at {SECRETS_PATH}")
        exit()
    try:
        with open(SECRETS_PATH, "rb") as f:
            secrets = tomllib.load(f)
            return secrets.get("TMDB_API_KEY")
    except Exception as e:
        print(f"❌ Error reading secrets: {e}")
        exit()

API_KEY = load_api_key()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Robust Fetcher ---
def get_session():
    """Creates a request session with auto-retries."""
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def fetch_top_movies():
    movies = []
    print(f"📡 Fetching top {PAGES_TO_FETCH} pages from TMDB...")
    
    base_url = "https://api.themoviedb.org/3/discover/movie"
    session = get_session()
    
    # Headers make us look like a real browser (Fixes 10054 errors)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }
    
    for page in range(1, PAGES_TO_FETCH + 1):
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "page": page,
            "vote_count.gte": 50
        }
        
        try:
            # Added timeout to prevent hanging
            r = session.get(base_url, params=params, headers=headers, timeout=10)
            
            if r.status_code == 200:
                data = r.json()
                for item in data['results']:
                    # Extract genre names
                    genre_ids = item.get('genre_ids', [])
                    genre_names = [GENRE_MAP.get(gid, '') for gid in genre_ids]
                    genres_str = ', '.join(filter(None, genre_names))
                    
                    movies.append({
                        "tmdbId": item['id'],
                        "title": item['title'],
                        "overview": item['overview'],
                        "release_date": item.get('release_date', ''),
                        "popularity": item['popularity'],
                        "vote_average": item['vote_average'],
                        "vote_count": item['vote_count'],
                        "poster_path": item.get('poster_path', ''),
                        "genres": genres_str
                    })
                print(f"✅ Page {page} done ({len(data['results'])} movies)")
            else:
                print(f"⚠️ Page {page} skipped (Status: {r.status_code})")
                
        except Exception as e:
            print(f"❌ Error on page {page}: {e}")
            
        time.sleep(0.5) # Increased sleep slightly to be nicer to the API
        
    return pd.DataFrame(movies)

# --- 3. Build Engine ---
def build_engine(df):
    if df.empty:
        print("❌ No movies fetched. Exiting.")
        return

    print(f"🧠 Generating Embeddings for {len(df)} movies...")
    df['text_content'] = df['title'] + ": " + df['overview'].fillna("")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text_content'].tolist(), show_progress_bar=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    print("💾 Saving models...")
    with open(f'{OUTPUT_DIR}/movies_metadata.pkl', 'wb') as f:
        pickle.dump(df, f)
    faiss.write_index(index, f"{OUTPUT_DIR}/faiss_index.bin")
    print("🎉 Success! Database Updated.")

if __name__ == "__main__":
    df = fetch_top_movies()
    build_engine(df)