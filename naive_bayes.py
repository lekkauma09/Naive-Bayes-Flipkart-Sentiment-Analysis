import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("Dataset-SA.csv")

# Display first five rows
print(df.head())

# Remove missing values
df = df.dropna()

# Combine Review and Summary columns
df["Text"] = df["Review"] + " " + df["Summary"]

# Features and Target
X = df["Text"]
y = df["Sentiment"]

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Naive Bayes model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict New Review
new_review = ["Excellent product with amazing quality and fast delivery"]

new_review_vector = vectorizer.transform(new_review)

prediction = model.predict(new_review_vector)

print("\nPredicted Sentiment:", prediction[0])
