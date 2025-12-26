import google.generativeai as genai
import streamlit as st
import os

class GeminiHandler:
    def __init__(self, api_key=None):
        self.client = None
        self.is_ready = False
        
        # 1. Try to get key from arguments
        final_key = api_key
        
        # 2. If no key provided, try Streamlit Secrets
        if not final_key:
            try:
                final_key = st.secrets["GOOGLE_API_KEY"]
            except:
                pass
        
        # 3. If still no key, try OS Environment Variables
        if not final_key:
            final_key = os.getenv("GOOGLE_API_KEY")

        # 4. Initialize Gemini
        if final_key:
            try:
                genai.configure(api_key=final_key)
                self.client = genai.GenerativeModel('gemini-2.5-flash')
                self.is_ready = True
            except Exception as e:
                print(f"Gemini Connection Error: {e}")
        else:
            print("⚠️ No Gemini API Key found. AI features disabled.")

    def generate_summary(self, user_query, rec_df):
        if not self.is_ready or rec_df.empty:
            return None

        movie_list_text = ""
        for i, row in rec_df.iterrows():
            genres = row.get('genres', 'N/A') if 'genres' in row else 'N/A'
            movie_list_text += f"{i+1}. {row['title']} (Genres: {genres})\n"

        prompt = f"""
        User Query: "{user_query}"
        Top Movies:
        {movie_list_text}

        Task: Write a 2-sentence summary explaining why these movies fit the vibe.
        """
        
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            return None