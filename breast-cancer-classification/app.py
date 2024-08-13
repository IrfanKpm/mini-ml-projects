import pickle
import numpy as np
import pandas as pd

# Load the saved model and scaler
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def predict(features):
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    
    features_df = pd.DataFrame([features], columns=feature_names)
    
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    return 'Malignant' if prediction[0] == 1 else 'Benign'

# Main script
if __name__ == "__main__":
    # Example input features (these should be replaced with real values for testing)
    example_features = [
        14.5, 23.5, 90.3, 600.0, 0.1, 0.1, 0.2, 0.3, 0.2, 0.06,
        1.0, 1.2, 7.0, 80.0, 0.03, 0.04, 0.1, 0.2, 0.03, 0.04,
        14.0, 25.0, 85.0, 550.0, 0.09, 0.12, 0.18, 0.28, 0.19, 0.055
    ]

    result = predict(example_features)
    print(f'\n\nThe prediction for the input features is: {result}\n')
