# ==========================================
# TASK 4 : FEATURE ENCODING & SCALING
# DATASET : Churn_Modelling.csv
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# STEP 1 : LOAD DATASET
# ==========================================

df = pd.read_csv("Churn_Modelling.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# STEP 2 : DROP UNNECESSARY COLUMNS
# ==========================================

df.drop(['RowNumber', 'CustomerId', 'Surname'],
        axis=1,
        inplace=True)

print("\nColumns After Dropping:")
print(df.columns)

# ==========================================
# STEP 3 : IDENTIFY CATEGORICAL VARIABLES
# ==========================================

print("\nCategorical Columns:")
print(df.select_dtypes(include='object').columns)

# ==========================================
# STEP 4 : ONE-HOT ENCODING
# ==========================================

df_encoded = pd.get_dummies(
    df,
    columns=['Geography', 'Gender'],
    drop_first=True
)

print("\nData After Encoding:")
print(df_encoded.head())

# ==========================================
# STEP 5 : FEATURE SCALING
# ==========================================

numerical_columns = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary'
]

# ==========================================
# STANDARDIZATION
# ==========================================

standard_scaler = StandardScaler()

df_standardized = df_encoded.copy()

df_standardized[numerical_columns] = standard_scaler.fit_transform(
    df_standardized[numerical_columns]
)

print("\nStandardized Data:")
print(df_standardized.head())

# ==========================================
# NORMALIZATION
# ==========================================

minmax_scaler = MinMaxScaler()

df_normalized = df_encoded.copy()

df_normalized[numerical_columns] = minmax_scaler.fit_transform(
    df_normalized[numerical_columns]
)

print("\nNormalized Data:")
print(df_normalized.head())

# ==========================================
# STEP 6 : HISTOGRAM BEFORE SCALING
# ==========================================

plt.figure(figsize=(8,5))
plt.hist(df_encoded['EstimatedSalary'], bins=30)
plt.title("Before Scaling")
plt.xlabel("Estimated Salary")
plt.ylabel("Frequency")
plt.show()

# ==========================================
# STEP 7 : HISTOGRAM AFTER STANDARDIZATION
# ==========================================

plt.figure(figsize=(8,5))
plt.hist(df_standardized['EstimatedSalary'], bins=30)
plt.title("After Standardization")
plt.xlabel("Estimated Salary")
plt.ylabel("Frequency")
plt.show()

# ==========================================
# STEP 8 : HISTOGRAM AFTER NORMALIZATION
# ==========================================

plt.figure(figsize=(8,5))
plt.hist(df_normalized['EstimatedSalary'], bins=30)
plt.title("After Normalization")
plt.xlabel("Estimated Salary")
plt.ylabel("Frequency")
plt.show()

# ==========================================
# STEP 9 : BOXPLOT COMPARISON
# ==========================================

plt.figure(figsize=(8,5))
plt.boxplot(df_encoded['EstimatedSalary'])
plt.title("Before Scaling Boxplot")
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot(df_standardized['EstimatedSalary'])
plt.title("After Standardization Boxplot")
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot(df_normalized['EstimatedSalary'])
plt.title("After Normalization Boxplot")
plt.show()

# ==========================================
# FINAL DATASET INFO
# ==========================================

print("\nFinal Dataset Shape:")
print(df_encoded.shape)

print("\nTask 4 Completed Successfully!")


