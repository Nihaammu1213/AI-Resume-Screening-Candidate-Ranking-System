import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text.strip()

# Function to rank resumes based on similarity to the job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    # FIX: Use the correct function name `cosine_similarity`
    similarity_scores = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return similarity_scores

# Streamlit UI
st.title("AI Resume Screening & Candidate Ranking System")

# Job Description Input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# Upload Resumes
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process and Rank Resumes
if uploaded_files and job_description:
    st.header("Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
    
    scores = rank_resumes(job_description, resumes)

    # Display results
    ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)
    
    st.subheader("Resume Ranking (Higher Score = Better Match)")
    for i, (file, score) in enumerate(ranked_resumes):
        st.write(f"{i+1}. {file.name} - Similarity Score: {score:.2f}")





