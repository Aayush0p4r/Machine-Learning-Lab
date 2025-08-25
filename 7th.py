# Spam Email Classification
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a sample dataset
data = {
    "text": [
        "Win a lottery now",
        "Hello, how are you?",
        "Claim free prize",
        "Meeting at 10 AM",
        "Congratulations, you won",
    ],
    "label": [1, 0, 1, 0, 1] # 1 = spam, 0 = ham
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))