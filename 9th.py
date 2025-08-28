# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset
data = pd.read_csv("./Raisin_Dataset.csv")
data.columns = data.columns.str.strip()  # Remove any leading/trailing whitespace from column names
data['Class'] = data['Class'].map({'Kecimen': 0, 'Besni': 1})  # Encode target variable

# Step 3: Split Features and Target
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 4: Standardize the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Build and Train the Nerual Network Model
model = MLPClassifier(
    hidden_layer_sizes = (16, 8),
    activation = 'relu',
    solver = 'sgd',
    learning_rate_init = 0.01,
    max_iter = 500,
    random_state = 42
)
model.fit(X_train, y_train)

# step 6: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))