import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true])
df = df.sample(frac=1)

# Use full text
X = df["text"]
y = df["label"]

# Split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
xtrain_vec = vectorizer.fit_transform(xtrain)
xtest_vec = vectorizer.transform(xtest)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(xtrain_vec, ytrain)

# Accuracy
print("Accuracy:", model.score(xtest_vec, ytest))

# Save
joblib.dump(model, "lr_model.jb")
joblib.dump(vectorizer, "vectorizer.jb")
