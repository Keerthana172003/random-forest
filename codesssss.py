import numpy as np# for numerical operations and array manipulations
import pandas as pd#for handling and processing structured data
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer#handles missing values in data
# Load data from CSV file
csv_file_path='pm2.5,toulene.csv'  # Replace with your CSV file path
data=pd.read_csv(csv_file_path) #stores in 'data' as dataframe using alias pd 

x=data[['Toulene']].values#converts dataframe to numpy arrays using attribute 'values'
y=data['PM2.5'].values#x is treated as dataframe and y is treated as data series

# Handle missing values in x and y(exception handling for NaN values)
imputer_x=SimpleImputer(strategy='mean') 
imputer_y=SimpleImputer(strategy='mean')
# calculates the mean of the given coloumn and predicts the misssing
# values using the simpleimputer class that belongs to sklearn.impute
# module.the values are stored in imputer_x instance.
x_imputed=imputer_x.fit_transform(x)
# fits imputer to data and transforms the data with the imputed values
y_imputed=imputer_y.fit_transform(y.reshape(-1, 1))
# -1 in reshape adjusts the no. of rows according to original data and 1 is no.of coloumns

# Perform linear regression
regression=LinearRegression()
# regression is the instance of class linear regression
regression.fit(x_imputed, y_imputed)
#fits the data to regression model

slope=regression.coef_[0][0]
#coef attribute of the instance is used to extract the values
intercept=regression.intercept_[0]
# slope is extracted using a 2D array and intercept is extracted using 1D array

# Calculate correlation coefficient
correlation_coef=np.corrcoef(x_imputed.squeeze(), y_imputed.squeeze())[0, 1]
# np.corrcoef-It takes 2 arrays and returns a correlation matrix and the values are extracted from there
#.squeeze-remove unnecessary single-dimensional dimensions to work with simpler arrays.
#[0,1]-the value of correlaton coeeficient from the matrix is extracted.i=0 and j=1

print("Slope:", slope)
print("Intercept:", intercept)
print("Correlation Coefficient:", correlation_coef)
