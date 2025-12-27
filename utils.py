import pickle
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

class MovieRecommender:
    def __init__(self, models_dir="models"):
        # Load Metadata
        with open(f'{models_dir}/movies_metadata.pkl', 'rb') as f:
            self.movies = pickle.load(f)
            
        # Load FAISS Index
        self.index = faiss.read_index(f"{models_dir}/faiss_index.bin")
        
        # Load Encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def hybrid_recommend(self, query, top_k=5):
        # 1. Semantic Search
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(query_vector, top_k * 10) #  50 candidates
    
        recommendations = []
        candidates_idx = indices[0]
        candidates_dist = distances[0]
        
        for i, idx in enumerate(candidates_idx):
            if idx == -1: continue
            
            movie_data = self.movies.iloc[idx]
            
            # Semantic Score (0 to 1)
            semantic_score = 1 / (1 + candidates_dist[i])
            
            # Quality Score (Normalize Vote Average 0-10 -> 0-1)
            # This replaces the SVD "Personalized Rating"
            quality_score = movie_data['vote_average'] / 10.0
            
            # Popularity Boost (Log scale to dampen huge blockbusters)
            pop_score = np.log1p(movie_data['popularity']) / 10.0
            
            # Final Hybrid Formula
            # 70% Relevance (Does it match the query?)
            # 20% Quality (Is it a good movie?)
            # 10% Popularity (Is it trending?)
            final_score = (0.7 * semantic_score) + (0.2 * quality_score) + (0.1 * pop_score)
            
            recommendations.append({
                "tmdbId": movie_data['tmdbId'],
                "title": movie_data['title'],
                "overview": movie_data['overview'],
                "release_date": movie_data['release_date'],
                "score": final_score,
                "rating": movie_data['vote_average'],
                "genres": movie_data.get('genres', 'N/A'),
                "poster_path": movie_data.get('poster_path', '')
            })
            
        # Sort and Return
        return pd.DataFrame(recommendations).sort_values(by="score", ascending=False).head(top_k)
