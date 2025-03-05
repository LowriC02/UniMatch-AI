import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
@st.cache_data
def load_data():
    courses_df = pd.read_csv("/mnt/data/KISCOURSE.csv")
    entry_df = pd.read_csv("/mnt/data/ENTRY.csv")
    employment_df = pd.read_csv("/mnt/data/EMPLOYMENT.csv")
    institutions_df = pd.read_csv("/mnt/data/INSTITUTION.csv")
    nss_df = pd.read_csv("/mnt/data/NSS.csv")
    
    # Merge datasets
    courses_df = courses_df.merge(entry_df, on=["PUBUKPRN", "KISCOURSEID", "KISMODE"], how="left")
    courses_df = courses_df.merge(employment_df, on=["PUBUKPRN", "KISCOURSEID", "KISMODE"], how="left")
    courses_df = courses_df.merge(institutions_df, on=["PUBUKPRN", "UKPRN"], how="left")
    courses_df = courses_df.merge(nss_df, on=["PUBUKPRN", "KISCOURSEID", "KISMODE"], how="left")
    
    return courses_df

courses_df = load_data()

# Function to recommend courses
def recommend_courses(student_interests, predicted_grades):
    student_interests = student_interests.lower()
    eligible_courses = courses_df.dropna(subset=['ENTRY'])
    eligible_courses = eligible_courses[eligible_courses['ENTRY'] <= predicted_grades]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    course_vectors = vectorizer.fit_transform(eligible_courses['TITLE'].fillna(''))
    student_vector = vectorizer.transform([student_interests])
    
    similarity_scores = cosine_similarity(student_vector, course_vectors).flatten()
    eligible_courses["Similarity"] = similarity_scores
    ranked_courses = eligible_courses.sort_values(by="Similarity", ascending=False)
    
    return ranked_courses[["TITLE", "UCASPROGID", "LEGAL_NAME", "PROVURL", "EMPLOYMENT"]].head(5)

# Streamlit UI
st.title("🎓 AI-Free UCAS Course Recommender")
st.write("Enter your preferences to find the best degree options for you!")

student_interests = st.text_area("What subjects or career areas are you interested in?")
predicted_grades = st.text_input("Enter your predicted grades (e.g., AAB, BBB, etc.)")

if st.button("Find My Course!"):
    if student_interests and predicted_grades:
        recommendations = recommend_courses(student_interests, predicted_grades)
        st.write("## 🔍 Recommended Courses")
        st.table(recommendations)
    else:
        st.warning("⚠️ Please enter your interests and grades to get recommendations.")
