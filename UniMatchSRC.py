import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
@st.cache_data
def load_data():
    try:
        courses_df = pd.read_csv("/mnt/data/KISCOURSE.csv")
        entry_df = pd.read_csv("/mnt/data/ENTRY.csv")
        employment_df = pd.read_csv("/mnt/data/EMPLOYMENT.csv")
        institutions_df = pd.read_csv("/mnt/data/INSTITUTION.csv")
        nss_df = pd.read_csv("/mnt/data/NSS.csv")
        
        # Ensure necessary columns exist
        required_columns = {"PUBUKPRN", "KISCOURSEID", "KISMODE", "TITLE", "ENTRY"}
        for df in [courses_df, entry_df, employment_df, institutions_df, nss_df]:
            if not required_columns.issubset(df.columns):
                st.error(f"Missing required columns in dataset: {df.columns}")
                return pd.DataFrame()

        # Merge datasets safely
        courses_df = courses_df.merge(entry_df, on=["PUBUKPRN", "KISCOURSEID", "KISMODE"], how="left")
        courses_df = courses_df.merge(employment_df, on=["PUBUKPRN", "KISCOURSEID", "KISMODE"], how="left")
        courses_df = courses_df.merge(institutions_df, on=["PUBUKPRN", "UKPRN"], how="left")
        courses_df = courses_df.merge(nss_df, on=["PUBUKPRN", "KISCOURSEID", "KISMODE"], how="left")
        
        return courses_df.fillna("")  # Fill missing values to avoid errors
    except FileNotFoundError as e:
        st.error(f"Error loading datasets: {e}")
        return pd.DataFrame()

courses_df = load_data()

# Function to recommend courses
def recommend_courses(student_interests, predicted_grades):
    if courses_df.empty:
        return pd.DataFrame()

    student_interests = student_interests.lower().strip()

    try:
        predicted_grades = int(predicted_grades)
    except ValueError:
        st.error("Please enter a valid numerical grade value.")
        return pd.DataFrame()

    # Ensure 'ENTRY' exists and is numeric
    if "ENTRY" not in courses_df.columns or courses_df["ENTRY"].dtype == object:
        st.error("ENTRY column is missing or non-numeric.")
        return pd.DataFrame()

    eligible_courses = courses_df.dropna(subset=['ENTRY'])
    eligible_courses = eligible_courses[eligible_courses['ENTRY'].astype(float) <= predicted_grades]

    if eligible_courses.empty:
        st.warning("No courses found matching your predicted grades.")
        return pd.DataFrame()

    # Ensure 'TITLE' is present
    eligible_courses["TITLE"] = eligible_courses["TITLE"].fillna("")
    if eligible_courses["TITLE"].isna().all():
        st.warning("No course titles available for comparison.")
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words='english')
    course_titles = eligible_courses['TITLE'].fillna('')
    course_vectors = vectorizer.fit_transform(course_titles)
    student_vector = vectorizer.transform([student_interests])

    similarity_scores = cosine_similarity(student_vector, course_vectors).flatten()
    eligible_courses["Similarity"] = similarity_scores
    ranked_courses = eligible_courses.sort_values(by="Similarity", ascending=False)

    return ranked_courses[["TITLE", "UCASPROGID", "LEGAL_NAME", "PROVURL", "EMPLOYMENT"]].head(5)

# Streamlit UI
st.title("🎓 AI-Free UCAS Course Recommender")
st.write("Enter your preferences to find the best degree options for you!")

student_interests = st.text_area("What subjects or career areas are you interested in?")
predicted_grades = st.text_input("Enter your predicted grades (e.g., 120, 100, etc.)")

if st.button("Find My Course!"):
    if student_interests and predicted_grades:
        recommendations = recommend_courses(student_interests, predicted_grades)
        if not recommendations.empty:
            st.write("## 🔍 Recommended Courses")
            st.table(recommendations)
    else:
        st.warning("⚠️ Please enter your interests and grades to get recommendations.")
