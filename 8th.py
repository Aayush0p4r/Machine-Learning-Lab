# Data Analysis using Raisin Dataset 
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("./Raisin_Dataset.csv")

# Encode target class
data['Class'] = data['Class'].map({'Kecimen': 0, 'Besni': 1})

# Split data 
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#scaler the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with (L1)
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

print("Lasso Accuracy:", accuracy_score(y_test, y_pred_lasso))

# Logistic Regression with Ridge (L2)
ridge_model = LogisticRegression(penalty='l2', solver='lbfgs', C=1, max_iter=1000)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

print("Ridge Accuracy:", accuracy_score(y_test, y_pred_ridge))