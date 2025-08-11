# Introduction to the language – Importing datasets – Data visualization.
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the iris dataset
iris = load_iris()

# Create a DAtaFrame from the iris dataset.
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first 5 rows of the DataFrame.
print("Iris Dataset:")
print(df.head())

# Pairplot for visualizing relationships between features
sns.pairplot(df, hue='target', diag_kind='kde', palette='bright')
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop('target', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Iris Features")
plt.show()

# Boxplot for each feature grouped by target class
plt.figure(figsize=(12, 8))

for i, column in enumerate(iris.feature_names, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='target', y=column, data=df, palette='Set2')
    plt.title(column)

plt.tight_layout()
plt.show()


