# Checking a student's suitability for a perticular activity 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a sample dataset
data = {
    "gpa": [7.5, 6.0, 8.2, 5.5, 9.0, 6.5, 7.8, 8.5],
    "coding_hourse": [10, 2, 15, 1, 20, 3, 12, 14],
    "suitable": [1, 0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[["gpa", "coding_hourse"]]
y = df["suitable"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))