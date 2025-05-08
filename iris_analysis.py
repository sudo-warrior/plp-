
# Iris Dataset Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].apply(lambda x: iris.target_names[x])

# Display first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())

# Dataset info
print("\nDataset Info:")
iris_df.info()

# Check for missing values
print("\nMissing Values:")
print(iris_df.isnull().sum())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(iris_df.describe())

# Group by species and compute means
print("\nMean values grouped by species:")
print(iris_df.groupby("species").mean(numeric_only=True))

# Set style for plots
sns.set(style="whitegrid")

# Line Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=iris_df.sort_values(by="species"), x=iris_df.index, y="petal length (cm)", hue="species")
plt.title("Petal Length by Sample Index")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("line_plot.png")
plt.clf()

# Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(data=iris_df, x="species", y="sepal width (cm)", estimator="mean")
plt.title("Average Sepal Width by Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Width (cm)")
plt.tight_layout()
plt.savefig("bar_plot.png")
plt.clf()

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(iris_df["petal length (cm)"], bins=20, kde=True)
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("hist_plot.png")
plt.clf()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=iris_df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.clf()

# Observations
print("""
Observations:
- Setosa has the smallest petal length and width, making it easily distinguishable.
- Virginica tends to have the largest overall measurements.
- Versicolor overlaps slightly with Virginica, particularly in petal dimensions.
""")
