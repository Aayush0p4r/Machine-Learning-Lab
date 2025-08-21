import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Read data from CSV file
data = pd.read_csv('house_features.csv')

# Assuming the CSV has columns: 'feature1', 'feature2', 'feature3', 'feature4', 'price'
# Adjust column names based on your actual CSV structure
X = data[['feature1', 'feature2', 'feature3', 'feature4']].values
y = data['price'].values

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Print results
print("Predicted Prices:", y_pred)
print("RMSE:", rmse)
print("R^2 Score:", r2)

# Plot Actual vs Predicted Prices
plt.scatter(y, y_pred, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()