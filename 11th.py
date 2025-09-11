# Naive Bayes Classifier Implementation
# Step 1: Import the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the CSV dataset
print("Step 1: Loading the dataset...")
data = pd.read_csv("PlayTennis.csv")
print("Dataset loaded Successfully.")
print(data.head())

# Step 3: Encode categorical variables
print("\nStep 2: Encoding categorical variables...")
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
print("Encoding Complpete.")
print(data.head())

# Step 3: Split into features and target
print("\nStep 3: Splitting into features (X) and targrt (y)...")
X = data.drop("Play Tennis", axis=1)
y = data["Play Tennis"]
print("Features and target prepared.")

# Step 4: Train-Test Split
print("\nStep 4: Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Train-test split complete.")

# Step 5: Train Naive Bayes Model
print("\nStep 5: Training the Naive Bayes model...")
model = GaussianNB()
model.fit(X_train, y_train)
print("Model training complete.")

# Step 6: Make Predictions
print("\nStep 6: Making predictions on the test set...")
y_pred = model.predict(X_test)
print("Predictions complete.")

# Step 7: Evaluate the Model
print("\nStep 7: Evaluating the model...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))