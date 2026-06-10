# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("BreastCancer.csv")

# Remove unnecessary column
df = df.drop("id", axis=1)

# Convert target values
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# Features (X) and Target (y)
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# Decision Tree
# ---------------------------
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy =", dt_accuracy)

# ---------------------------
# Random Forest
# ---------------------------
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy =", rf_accuracy)

# ---------------------------
# Feature Importance
# ---------------------------
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

importance.nlargest(10).plot(kind="barh")

plt.title("Top 10 Important Features")
plt.show()
