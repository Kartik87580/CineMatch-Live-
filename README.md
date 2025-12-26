
# 🎬 CineMatch Live: AI-Powered Movie Recommender

**CineMatch Live** is an advanced hybrid movie recommendation system that combines **Semantic Search (Vector Embeddings)** with **Real-time TMDB Data** and **Generative AI (Google Gemini)** to provide personalized, context-aware movie suggestions.

unlike traditional recommenders that rely on static datasets, CineMatch Live crawls the latest movie data, understands natural language queries (e.g., *"A sad sci-fi movie about isolation"*), and explains *why* a movie was recommended using an LLM.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/314dcbf1-7f36-4565-ba15-11aba3017d9e" />


---

## 🚀 Key Features

* **🧠 Hybrid Recommendation Engine:** * **Semantic Search:** Uses `SentenceTransformers` (all-MiniLM-L6-v2) to convert movie plots into 384-dimensional vectors.
    * **FAISS Vector DB:** Enables lightning-fast similarity search across thousands of movies.
    * **Weighted Scoring:** Combines *Semantic Relevance* (70%), *TMDB Rating* (20%), and *Popularity* (10%) for the perfect rank.
* **🤖 Generative AI Integration:** * Uses **Google Gemini 1.5 Flash** to act as a "Movie Concierge," summarizing the vibe of the recommendations and explaining why they fit the user's mood.
* **📡 Live Data Pipeline:** * Custom Python crawler (`build_tmdb_dataset.py`) fetches the latest trending movies (2024-2025) directly from the TMDB API.
    * Auto-retries on connection failures to ensure robust data collection.
* **🎨 Interactive UI:** * Built with **Streamlit**.
    * Displays high-quality movie posters.
    * Fast, responsive interface with cached models.

---

## 🛠️ Tech Stack

| Component | Technology Used |
| :--- | :--- |
| **Frontend** | Streamlit (Python) |
| **LLM / AI** | Google Gemini API (1.5 Flash) |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | Sentence-Transformers (HuggingFace) |
| **Data Source** | TMDB API (The Movie Database) |
| **Data Processing** | Pandas, NumPy |

---

## 📂 Project Structure

```text
MRS_V_2.0/
│
├── .streamlit/
│   └── secrets.toml       # API Keys (TMDB + Google Gemini)
│
├── data/                  # Raw fetched data (Auto-generated)
├── models/                # Trained models & Vector Indices
│   ├── faiss_index.bin    # The FAISS vector database
│   └── movies_metadata.pkl# DataFrame with titles, plots, posters
│
├── app.py                 # Main Streamlit Application (Frontend)
├── build_tmdb_dataset.py  # Data Pipeline (Fetcher + Vectorizer)
├── utils.py               # Core Hybrid Recommendation Logic
├── llm_utils.py           # Gemini AI Handler
└── requirements.txt       # Project Dependencies

```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/Kartik87580/CineMatch-Live-
cd CineMatch-Live-

```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure API Keys

Create a file `.streamlit/secrets.toml` and add your keys:

```toml
GOOGLE_API_KEY = "your_google_gemini_key"
TMDB_API_KEY = "your_tmdb_api_key"

```

### 5. Build the Dataset (Run Once)

This script crawls TMDB, generates embeddings, and saves the models locally.

```bash
python build_tmdb_dataset.py

```

*Wait for the "🎉 Success!" message.*

### 6. Run the App

```bash
streamlit run app.py

```

---


---

## 🧠 How It Works

1. **Ingestion:** `build_tmdb_dataset.py` fetches top movies from TMDB. It combines the `Title` + `Overview` into a single text block.
2. **Vectorization:** The text is passed through a **BERT-based model** to create dense vector embeddings. These are stored in a FAISS index.
3. **Retrieval:** When a user types a query (e.g., *"Inspiring sports movie"*), the query is vectorized and compared against the FAISS index using Cosine Similarity.
4. **Ranking:** The top 50 semantic matches are re-ranked using a weighted formula that boosts movies with higher **TMDB ratings** and **Popularity**.
5. **Explanation:** The top 5 results are sent to **Gemini AI**, which generates a natural language summary explaining the selection.

---

## 🔮 Future Improvements

* [ ] **Filter by Genre/Year:** Add sidebar filters to narrow down search results.
* [ ] **"More Like This":** Click a movie to find similar ones (Item-to-Item filtering).
* [ ] **User History:** Implement session state to "remember" what the user liked during the session.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

---

**Author:** [Your Name]

**License:** MIT

```

```
