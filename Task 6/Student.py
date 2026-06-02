# ==========================================================
# TASK 6: DATA CLEANING & PREPROCESSING
# AI & ML Internship
# Dataset: Telecom Churn Dataset
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# ==========================================================
# STEP 1: LOAD DATASET
# ==========================================================

df = pd.read_csv("telecom_churn.csv")

print("="*50)
print("DATASET LOADED SUCCESSFULLY")
print("="*50)

print("\nFirst 5 Records:")
print(df.head())

# ==========================================================
# STEP 2: DATASET INFORMATION
# ==========================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================================
# STEP 3: CHECK MISSING VALUES
# ==========================================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================================
# STEP 4: CHECK DUPLICATES
# ==========================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:", duplicate_count)

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:")
print(df.shape)

# ==========================================================
# STEP 5: ENCODE CATEGORICAL FEATURES
# ==========================================================

print("\nEncoding Categorical Columns...")

categorical_columns = df.select_dtypes(include='object').columns

le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

print("Categorical Encoding Completed")

# ==========================================================
# STEP 6: OUTLIER DETECTION
# ==========================================================

numeric_columns = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15,6))
sns.boxplot(data=df[numeric_columns])
plt.xticks(rotation=90)
plt.title("Boxplot Before Outlier Treatment")
plt.show()

# ==========================================================
# STEP 7: OUTLIER TREATMENT USING IQR
# ==========================================================

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df[col] = np.where(
        df[col] > upper_limit,
        upper_limit,
        np.where(
            df[col] < lower_limit,
            lower_limit,
            df[col]
        )
    )

print("\nOutlier Treatment Completed")

# ==========================================================
# STEP 8: FEATURE ENGINEERING
# ==========================================================

if 'date_of_registration' in df.columns:

    df['date_of_registration'] = pd.to_datetime(
        df['date_of_registration'],
        errors='coerce'
    )

    df['registration_year'] = df['date_of_registration'].dt.year
    df['registration_month'] = df['date_of_registration'].dt.month

    df.drop('date_of_registration', axis=1, inplace=True)

print("\nFeature Engineering Completed")

# ==========================================================
# STEP 9: FEATURE SCALING
# ==========================================================

numeric_columns = df.select_dtypes(include=np.number).columns

scaler = StandardScaler()

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nFeature Scaling Completed")

# ==========================================================
# STEP 10: FINAL DATASET
# ==========================================================

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Dataset Preview:")
print(df.head())

# ==========================================================
# STEP 11: SAVE PREPROCESSED DATASET
# ==========================================================

df.to_csv("preprocessed_dataset.csv", index=False)

print("\nPreprocessed Dataset Saved Successfully")

# ==========================================================
# STEP 12: FINAL VERIFICATION
# ==========================================================

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nFinal Dataset Information:")
print(df.info())

print("\nTASK 6 COMPLETED SUCCESSFULLY")
