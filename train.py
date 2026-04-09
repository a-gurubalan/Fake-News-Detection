import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data from CSV files
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = 0  # Fake news
true_df["label"] = 1  # Real news

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Use the 'text' column for features
X = df["text"]
y = df["label"]

# Vectorize the text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vec, y)

# Save the model and vectorizer
joblib.dump(model, "lr_model.jb")
joblib.dump(vectorizer, "vectorizer.jb")

print("Model trained and saved successfully")