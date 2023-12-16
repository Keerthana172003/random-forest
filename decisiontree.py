import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_regression  # Sample dataset generator

# Assuming your data is in a file named 'your_data.csv'
file_path = 'BTM_hourlyy2.csv'

# Load your data into a Pandas DataFrame
df = pd.read_csv(file_path)
df=df.apply(pd.to_numeric,errors='coerce')

# Replace 'None' values with actual NaN values for easier handling
df.replace('None', pd.NA, inplace=True)

# Separating features and target variable
X = df.drop('PM2.5', axis=1)  # Features
y = df['PM2.5']  # Target variable

# Creating SimpleImputer instance to impute missing values with the mean
imputer = SimpleImputer(strategy='most_frequent')

# Creating SimpleImputer instance to impute missing values with the mean for y
imputer_y = SimpleImputer(strategy='most_frequent')

# Impute missing values in the features
X_imputed = imputer.fit_transform(X)

# Reshape y to a 2D array
y = y.values.reshape(-1, 1)
y_imputed = imputer_y.fit_transform(y)

# Generating sample dataset (you can replace this with your own dataset)
X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.4, random_state=42)

# Create the decision tree regressor
tree_reg = DecisionTreeRegressor(random_state=42)

# Fit the model on the training data
tree_reg.fit(X_train, y_train)

# Predict on the test set
predictions = tree_reg.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
# Predict on the test set
predictions = tree_reg.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, predictions)
print(f"R-squared: {r2}")
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Create a DataFrame with actual and predicted values
results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})
print(results.head())  # Display the first few rows of the DataFrame

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to search
param_grid = {
    'max_depth': [3, 5, 7, 9],  # example values, modify as needed
    'min_samples_split': [2, 5, 10]
}

# Create GridSearchCV object
grid_search = GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get best hyperparameters
best_params = grid_search.best_params_

# Train a new model using the best hyperparameters
best_tree_reg = DecisionTreeRegressor(**best_params)
best_tree_reg.fit(X_train, y_train)

# Evaluate the new model
best_predictions = best_tree_reg.predict(X_test)
best_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
print(f"Best RMSE after tuning: {best_rmse}")

import matplotlib.pyplot as plt
# RMSE values before and after tuning
rmse_values = [rmse, best_rmse]  # Replace with your actual RMSE values

# Labels for the bars
labels = ['Before Tuning', 'After Tuning']

# Create bar plot
plt.figure(figsize=(8, 6))
plt.bar(labels, rmse_values, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE Before and After Hyperparameter Tuning')
plt.show()