import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Text cleaning + normalization
# -------------------------------
def clean_text(text):
    text = str(text).lower()

    # normalize common terms
    text = text.replace("ml", "machine learning")
    text = text.replace("ai", "artificial intelligence")

    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("resume_job_dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna()
    return df

df = load_data()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Resume Analyzer", layout="centered")

st.title("📄 Resume Analyzer with Job Role")

# Dropdown
roles = df["Job_Role"].unique()
selected_role = st.selectbox("Select Job Role", roles)

# Get job description
job_desc = df[df["Job_Role"] == selected_role]["Job_Description"].iloc[0]

# Display job description
st.text_area("Job Description", job_desc, height=120)

# Resume input
resume_input = st.text_area("Enter Resume Text")

# -------------------------------
# Analyze
# -------------------------------
if st.button("Analyze"):

    if resume_input:

        # Clean text
        resume_clean = clean_text(resume_input)
        job_clean = clean_text(job_desc)

        # 🔥 Combine dataset + input (IMPORTANT FIX)
        temp_df = df.copy()
        temp_df.loc[len(temp_df)] = [0, selected_role, resume_clean, job_clean, 0]

        # Combine text
        temp_df["combined"] = temp_df["Resume"] + " " + temp_df["Job_Description"]

        # TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(temp_df["combined"])

        # Compare last row with dataset
        similarity = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])

        score = similarity.max()
        score_percent = round(score * 100, 2)

        # Output
        st.subheader("🔍 Match Score")
        st.write(f"Match: **{score_percent}%**")

        if score_percent > 80:
            st.success("Excellent Match ✅")
        elif score_percent > 50:
            st.warning("Moderate Match ⚠️ Improve skills")
        else:
            st.error("Low Match ❌ Add relevant keywords")

    else:
        st.warning("Please enter resume text")