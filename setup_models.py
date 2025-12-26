import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import pickle
import faiss
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

# 1. Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def download_data():
    print("⬇️ Downloading MovieLens dataset...")
    # Using a slightly older, stable link just in case
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("data")
        print("✅ Data downloaded.")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        exit()

def build_models():
    print("📊 Loading and Merging Data...")
    movies_df = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings_df = pd.read_csv("data/ml-latest-small/ratings.csv")
    # NEW: Load links to get TMDB ID for posters
    links_df = pd.read_csv("data/ml-latest-small/links.csv")

    # NEW: Merge movies with links, keeping all movies (left join)
    movies = movies_df.merge(links_df, on='movieId', how='left')
    
    # Fill missing TMDB Ids with 0 and ensure it's an integer column
    movies['tmdbId'] = movies['tmdbId'].fillna(0).astype(int)
    print(f"Data shape after merge: {movies.shape}")
    
    # --- Part A: Collaborative Filtering (SVD) ---
    print("🤖 Training SVD Model (Scikit-Learn)...")
    user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = svd.fit_transform(user_movie_matrix)
    
    svd_data = {
        "user_matrix": matrix_reduced,
        "components": svd.components_,
        "user_ids": user_movie_matrix.index.tolist(),
        "movie_ids": user_movie_matrix.columns.tolist()
    }

    # --- Part B: Content-Based (Semantic Search) ---
    print("🧠 Generating Embeddings...")
    movies['overview'] = movies['title'] + " " + movies['genres'].str.replace('|', ' ')
    bert = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = bert.encode(movies['overview'].tolist(), show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # --- Part C: Save Everything ---
    print("💾 Saving models to ./models/ ...")
    # IMPORTANT: This pickle now contains 'tmdbId' column
    with open('models/movies_metadata.pkl', 'wb') as f:
        pickle.dump(movies, f)
        
    with open('models/svd_model.pkl', 'wb') as f:
        pickle.dump(svd_data, f)
        
    faiss.write_index(index, "models/faiss_index.bin")
    
    print("🎉 Setup Complete! The new metadata includes TMDB IDs for posters.")

if __name__ == "__main__":
    # Check if data exists before downloading again
    if not os.path.exists("data/ml-latest-small/movies.csv"):
        download_data()
    build_models()