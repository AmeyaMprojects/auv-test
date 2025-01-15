import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Fetch the Boston housing dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Extract features and target from the raw data
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
boston_df = pd.DataFrame(data, columns=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
])
boston_df['MEDV'] = target

# Exploratory Data Analysis: Plot the distribution of MEDV
plt.figure(figsize=(8, 6))
plt.hist(boston_df['MEDV'], bins=30, edgecolor='k')
plt.title('Distribution of MEDV (Median value of homes in $1000s)')
plt.xlabel('MEDV ($1000s)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Check for missing values
print("Missing values summary:\n", boston_df.isnull().sum())

# Preprocess: Handle missing values by replacing with the mean
imputer = SimpleImputer(strategy='mean')
boston_imputed = imputer.fit_transform(boston_df)

# Convert the imputed data back to DataFrame
boston_cleaned_df = pd.DataFrame(boston_imputed, columns=boston_df.columns)

# Split the data into training and testing sets (4:1 ratio)
X = boston_cleaned_df.drop('MEDV', axis=1)
y = boston_cleaned_df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print("Training set shape (X_train):", X_train.shape)
print("Testing set shape (X_test):", X_test.shape)
print("Training set shape (y_train):", y_train.shape)
print("Testing set shape (y_test):", y_test.shape)
