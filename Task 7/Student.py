# ================================
# Task 7 - Exploratory Data Analysis
# Edutech Solution AI & ML Internship
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("automobile.csv")

# -------------------------------
# Basic Information
# -------------------------------
print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# -------------------------------
# Data Cleaning
# -------------------------------
df = df.drop_duplicates()

for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# -------------------------------
# Univariate Analysis
# -------------------------------
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# -------------------------------
# Boxplots for Outlier Detection
# -------------------------------
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# -------------------------------
# Bivariate Analysis
# -------------------------------
if len(num_cols) >= 2:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df[num_cols[0]],
                    y=df[num_cols[1]])
    plt.title("Scatter Plot")
    plt.show()

# -------------------------------
# Correlation Heatmap
# -------------------------------
plt.figure(figsize=(12,8))
corr = df[num_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Pairplot
# -------------------------------
sns.pairplot(df[num_cols])
plt.show()

# -------------------------------
# Top Correlations
# -------------------------------
corr_matrix = df[num_cols].corr()

corr_pairs = corr_matrix.unstack()
corr_pairs = corr_pairs.sort_values(kind="quicksort")

print("\nTop Correlations:")
print(corr_pairs)

# -------------------------------
# Insights
# -------------------------------
print("\nEDA Completed Successfully")
