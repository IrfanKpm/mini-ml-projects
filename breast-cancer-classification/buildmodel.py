import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pickle

# Load the breast cancer dataset from sklearn
cancer_data = load_breast_cancer()
data = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)
data['diagnosis'] = cancer_data.target

# Display the value counts for the target variable
print(f"value counts: \n{data['diagnosis'].value_counts()}")

# Define features and target variable
X = data.drop('diagnosis', axis=1)
Y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize and fit the StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test = scaler.transform(X_test)        # Transform on testing data

# Initialize and train the model
model = LogisticRegression(solver='liblinear', max_iter=2000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)  # Predictions on training data

# Evaluate the model
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model to a pickle file
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the scaler to a pickle file
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print(f'\n\nTraining Accuracy: {accuracy_train}')
print(f'Testing Accuracy: {accuracy_test}')
print("\n\nModel saved to 'logistic_regression_model.pkl'")
print("Scaler saved to 'scaler.pkl'")
