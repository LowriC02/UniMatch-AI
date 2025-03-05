import os
import sys

# Ensure dependencies are installed
try:
    import openai
    import streamlit as st
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError as e:
    missing_module = str(e).split("'")[1]
    print(f"⚠️ {missing_module} not found. Installing it now...")
    os.system(f"{sys.executable} -m pip install {missing_module}")
    import openai  # Re-attempt import

# Set OpenAI API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
