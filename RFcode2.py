import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

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
imputer = SimpleImputer(strategy='mean')

# Creating SimpleImputer instance to impute missing values with the mean for y
imputer_y = SimpleImputer(strategy='mean')

# Impute missing values in the features
X_imputed = imputer.fit_transform(X)

# Reshape y to a 2D array
y = y.values.reshape(-1, 1)
y_imputed = imputer_y.fit_transform(y)


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the model
accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy}")
