import streamlit as st
import joblib
import re
import numpy as np
from scipy.special import expit  # Sigmoid function

# Page config
st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🛡️",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Cleaning function (must match training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Sidebar
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔍 Detection System", "📊 Model Summary"]
)

st.sidebar.divider()
st.sidebar.write("Developed as a Mini Project")
st.sidebar.write("Abhinanad K")
st.sidebar.write("Edwin James")
st.sidebar.write("Joyel Emmanual Glen")
st.sidebar.write("B.Tech CSE (Data Science)")

# Main UI
if page == "🔍 Detection System":
    st.title("🛡️ Fake Job Listing Detection System")
    st.markdown("Analyze job postings to detect potential fraud.")

    st.divider()

    job_description = st.text_area(
        "Enter Job Description",
        height=250,
        placeholder="Paste the complete job description here..."
    )

    if st.button("Analyze Job Posting", use_container_width=True):

        if job_description.strip() == "":
            st.warning("⚠ Please enter a job description before analyzing.")
        else:
            with st.spinner("Analyzing job posting..."):
                
                cleaned_input = clean_text(job_description)
                vectorized_input = vectorizer.transform([cleaned_input])

                # Prediction
                prediction = model.predict(vectorized_input)[0]

                # Decision score
                decision_score = model.decision_function(vectorized_input)[0]
                # Convert to fake_probability using sigmoid
                fake_probability = expit(decision_score)

                # Custom threshold (more strict) 
                if fake_probability >= 0.55: #55% Threshold
                    prediction = 1  # Treat as FAKE
                else:
                    prediction = 0  # Treat as REAL

            st.divider()

            # GIF URLs
            real_gif = "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif"  
            fake_gif = "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExb3h1NWFnZDRmMmMxd3VsZGE3MzVndWprYzRmdzA0em94eHFhY2psdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4pVtP5MvTTwi0EmtkW/giphy.gif"  

            # Display result
            if prediction == 1:
                st.error("🚨 This job posting is likely FAKE.")
                st.image(fake_gif, width=250)
                st.metric("Fraud Probability", f"{fake_probability * 100:.2f}%")
            else:
                st.success("✅ This job posting is likely REAL.")
                st.image(real_gif, width=250)
                st.metric("Real Job Confidence", f"{(1 - fake_probability) * 100:.2f}%")
            
elif page == "📊 Model Summary":

    st.title("📊 Model Summary")

    st.subheader("Model Information")
    st.write("• Algorithm: Support Vector Machine (LinearSVC)")
    st.write("• Feature Extraction: TF-IDF (Unigram + Bigram)")
    st.write("• Max Features: 8000")
    st.write("• Class Handling: Balanced Class Weights")

    st.divider()

    st.subheader("Dataset Information")
    st.write("• Original Dataset: ~17,880 job postings")
    st.write("• Augmented Samples: 60 synthetic modern scam samples")
    st.write("• Final Dataset Size: 17,939 records")

    st.divider()

    st.subheader("Model Performance")
    st.write("• Accuracy: ~97.7%")
    st.write("• Fake Class Recall: 83%")
    st.write("• Fake Class Precision: 76%")
    st.write("• Threshold Adjusted to 55% for better fraud sensitivity")

    st.divider()

    st.subheader("Key Improvements")
    st.write("• Added bigram support for phrase detection")
    st.write("• Performed targeted data augmentation")
    st.write("• Tuned decision threshold for fraud detection")
    st.write("• Integrated probability-based UI output")
