# Data-Collection-Data-Source-Identification
import pandas as pd

# Load dataset
df = pd.read_csv('Titanic-Dataset.csv')

# First 5 rows
print(df.head())

# Shape
print(df.shape)

# Data types
print(df.dtypes)

# Dataset info
print(df.info())

# Statistical summary
print(df.describe())

# Missing values
print(df.isnull().sum())

# Numerical columns
print(df.select_dtypes(include=['int64', 'float64']).columns)

# Categorical columns
print(df.select_dtypes(include=['object']).columns)

# Unique values
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].unique()[:10])


  
