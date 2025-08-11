# Introduction to the language – Importing datasets – Data visualization.

from sklearn.datasets import load_iris
import pandas as pd

#load the iris dataset
iris = load_iris()

# Create a DAtaFrame from the iris dataset.
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first 5 rows of the DataFrame.
print("Iris Dataset:")
print(df.head())