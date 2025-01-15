import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
boston_df = pd.DataFrame(data, columns=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
])
boston_df['MEDV'] = target

plt.figure(figsize=(8, 6))
plt.hist(boston_df['MEDV'], bins=30, edgecolor='k')
plt.title('Distribution of MEDV (Median value of homes in $1000s)')
plt.xlabel('MEDV ($1000s)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print("Missing values summary:\n", boston_df.isnull().sum())

imputer = SimpleImputer(strategy='mean')
boston_imputed = imputer.fit_transform(boston_df)

boston_cleaned_df = pd.DataFrame(boston_imputed, columns=boston_df.columns)

X = boston_cleaned_df.drop('MEDV', axis=1)
y = boston_cleaned_df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape (X_train):", X_train.shape)
print("Testing set shape (X_test):", X_test.shape)
print("Training set shape (y_train):", y_train.shape)
print("Testing set shape (y_test):", y_test.shape)
