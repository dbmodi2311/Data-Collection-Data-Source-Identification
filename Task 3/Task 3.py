import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET
iris = sns.load_dataset('iris')

# FIRST 5 ROWS
print("First 5 Rows Of Dataset:")
print(iris.head())

# DATASET INFORMATION
print("\nDataset Information:")
print(iris.info())

# MISSING VALUES
print("\nMissing Values:")
print(iris.isnull().sum())

# STATISTICAL ANALYSIS
print("\nStatistical Analysis:")
print(iris.describe())

# HISTOGRAM OF SEPAL LENGTH
plt.figure(figsize=(8,5))

sns.histplot(iris['sepal_length'])

plt.title("Histogram Of Sepal Length")

plt.show()

# HISTOGRAM OF SEPAL WIDTH
plt.figure(figsize=(8,5))

sns.histplot(iris['sepal_width'])

plt.title("Histogram Of Sepal Width")

plt.show()

# SCATTER PLOT
plt.figure(figsize=(8,5))

sns.scatterplot(
    x='sepal_length',
    y='sepal_width',
    data=iris
)

plt.title("Sepal Length Vs Sepal Width")

plt.show()

# BAR PLOT
plt.figure(figsize=(8,5))

sns.barplot(
    x='species',
    y='petal_length',
    data=iris
)

plt.title("Bar Plot of Species Vs Petal Length")

plt.show()

# CORRELATION HEATMAP
correlation = iris.drop('species', axis=1).corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# PAIRPLOT
sns.pairplot(iris, hue='species')

plt.show()

# FINAL INSIGHTS
print("\nFinal Insight:")

print("1. Dataset has no missing values.")
print("2. Petal length and petal width are highly correlated.")
print("3. Setosa species is clearly separated.")
print("4. Some outliers are visible in sepal width.")
