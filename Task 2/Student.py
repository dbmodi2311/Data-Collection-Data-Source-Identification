import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/content/Titanic-Dataset.csv')

# Display first 5 rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Heatmap for missing values
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Fill missing Age values using Median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Fare values using Mean
df['Fare'].fillna(df['Fare'].mean(), inplace=True)

# Fill missing Embarked values using Mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column
df.drop('Cabin', axis=1, inplace=True)

# Verify missing values after cleaning
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv('Cleaned_Titanic_Dataset.csv', index=False)

print("Dataset cleaned successfully")
