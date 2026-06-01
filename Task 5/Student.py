# Task 5: Train-Test Split & Evaluation Metrics
# Heart Disease Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Load Dataset
df = pd.read_csv("heart.csv")

# Display Dataset Information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Convert Categorical Columns into Numeric
X = pd.get_dummies(X, drop_first=True)

# Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Create Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Results
print("\n----- Model Evaluation -----")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

print("\nConfusion Matrix:")
print(cm)

# Interpretation
print("\n----- Interpretation -----")
print("Accuracy shows overall correctness of the model.")
print("Precision shows how many predicted positive cases were actually positive.")
print("Recall shows how many actual positive cases were correctly identified.")
print("Confusion Matrix displays TP, TN, FP, and FN values.")
