import streamlit as st
import pandas as pd
import openai
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

# AI API Configuration
openai.api_key = st.secrets["OPENAI_API_KEY"]

def ai_enhanced_recommendation(student_interests, predicted_grades):
    """
    Uses AI to enhance course recommendations by summarising key details.
    """
    base_recommendations = recommend_courses(student_interests, predicted_grades)
    
    if base_recommendations.empty:
        return "No suitable courses found. Try adjusting your inputs."
    
    course_list = "\n".join([f"{row['TITLE']} at {row['LEGAL_NAME']}" for _, row in base_recommendations.iterrows()])
    
    prompt = f"""
    Given the following university courses:
    {course_list}
    
    The user is interested in: {student_interests}
    
    Generate a brief, personalised recommendation explaining why these courses are a good fit.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI course advisor."},
                  {"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

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
st.title("🎓 AI-Powered UCAS Course Recommender")
st.write("Enter your preferences to find the best degree options for you!")

student_interests = st.text_area("What subjects or career areas are you interested in?")
predicted_grades = st.text_input("Enter your predicted grades (e.g., AAB, BBB, etc.)")

if st.button("Find My Course!"):
    if student_interests and predicted_grades:
        ai_summary = ai_enhanced_recommendation(student_interests, predicted_grades)
        recommendations = recommend_courses(student_interests, predicted_grades)
        st.write("## 🔍 AI-Enhanced Course Recommendations")
        st.table(recommendations)
        st.write("### 🤖 AI Advice")
        st.write(ai_summary)
    else:
        st.warning("⚠️ Please enter your interests and grades to get recommendations.")
