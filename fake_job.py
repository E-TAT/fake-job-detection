import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('fake_job_postings.csv')
df = df[['description', 'fraudulent']]
df = df.dropna(subset=['description'])

# Preprocessing
df['description'] = df['description'].str.lower()
df['description'] = df['description'].str.replace('\n', ' ', regex=False)
df['description'] = df['description'].str.replace('\r', ' ', regex=False)

def remove_html(text):
    return re.sub(r'<.*?>', ' ', text)

def clean_text(text):
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['description'] = df['description'].apply(remove_html)
df['description'] = df['description'].apply(clean_text)

# Synthetic Fake Job Data Augmentation
synthetic_fake_jobs = []

base_templates = [
    "We are hiring remote assistants to manage cryptocurrency transfers for international clients. No prior experience required. You must complete wallet activation by sending a small crypto payment. Immediate approval guaranteed.",
    
    "Urgent hiring for online payment processing agents. Earn up to $7000 per month. A refundable security deposit is required to activate your employee dashboard.",
    
    "Work from home opportunity. Earn $5000 weekly with flexible hours. A one-time registration fee is required before onboarding.",
    
    "International digital finance company hiring transfer coordinators. Applicants must verify their account by paying a small activation charge.",
    
    "Crypto exchange platform hiring remote verification staff. Salary starts at $6000 monthly. Initial crypto deposit required for account setup.",
    
    "Remote job opportunity processing overseas transactions. No experience needed. Security fee required to unlock company portal.",
    
    "Global investment firm hiring online account managers. Quick hiring process. Registration payment required before access.",
    
    "Online logistics company seeking fund transfer assistants. Must submit refundable wallet verification payment before starting.",
    
    "Earn high income from home by handling digital transfers. Verification deposit required before employment confirmation.",
    
    "Immediate hiring. Process international payments remotely. Activation fee required to begin work."
]

# Duplicate templates with small variations
for i in range(6):  # 6 rounds × 10 templates = 60 samples
    for template in base_templates:
        synthetic_fake_jobs.append(template + f" Position batch ID {i}.")

synthetic_df = pd.DataFrame({
    "description": synthetic_fake_jobs,
    "fraudulent": [1] * len(synthetic_fake_jobs)
})

df = pd.concat([df, synthetic_df], ignore_index=True)

print("New dataset size after augmentation:", df.shape)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=8000,
    ngram_range=(1, 2),   # 🔥 Unigrams + Bigrams
    min_df=2
)
X = vectorizer.fit_transform(df['description'])
y = df['fraudulent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balanced SVM
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("Balanced SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Interpretability
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
top_fake_indices = coefficients.argsort()[-20:]
top_fake_words = [feature_names[i] for i in top_fake_indices]

print("\nTop words indicating FAKE jobs:")
print(top_fake_words)

# Save model
joblib.dump(model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')