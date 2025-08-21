# Linear Regression Example
# Training and Testing a Linear Regression Model
# Using sklearn to fit a model and evaluate its performance

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data
X = np.array([600, 800, 1000, 1200, 1500]).reshape(-1, 1)
y = np.array([28, 35, 45, 52, 65])

# Split into training & testing sets
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Train model
model = LinearRegression()
model.fit(Xtr, ytr)

# Parameters
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Predictions & Metrics
y_pred = model.predict(Xte)
print("MAE:", mean_absolute_error(yte, y_pred))
print("RMSE:", mean_squared_error(yte, y_pred))
print("R^2:", r2_score(yte, y_pred))

# Visualization
xx = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
yy = model.predict(xx)
plt.scatter(X, y, label = "Data")
plt.plot(xx, yy, label = "sklearn fit")
plt.xlabel("Area (sqft)")
plt.ylabel("Price ( Lakh)")
plt.legend() ; plt.title("Simpl Linear Regression (sklearn)")
plt.show()