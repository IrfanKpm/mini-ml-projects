import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# URL of the CSV file
url = "https://raw.githubusercontent.com/IrfanKpm/Machine-Learning-Notes1/main/datasets/heart.csv"

# Load the dataset
data = pd.read_csv(url)

# Display basic information about the dataset
print(f"Shape of the data: {data.shape}")
print("Data info:")
data.info()
print("\nCheck for missing values:\n", data.isnull().sum())
print("\nStatistical summary:\n", data.describe())

# Splitting the data
X = data.drop('target', axis=1)  # Features (drop the target column)
y = data['target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1500)
model.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = model.predict(X_train)

# Make predictions on the testing data
y_test_pred = model.predict(X_test)

# Calculate accuracy on training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Calculate accuracy on testing data
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Save the trained model as a .pkl file
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'logistic_regression_model.pkl'")


