# Breast Cancer Classification

This project uses a Logistic Regression model to classify breast cancer as either benign or malignant based on input features derived from the Breast Cancer Wisconsin dataset.

## Project Structure

- **`train_model.py`**: Script to train the logistic regression model and save the model and scaler.
- **`app.py`**: Script to load the saved model and scaler, take input features, and predict whether the breast cancer is benign or malignant.
- **`logistic_regression_model.pkl`**: Saved logistic regression model.
- **`scaler.pkl`**: Saved `StandardScaler` used to scale the input features.

### Make Predictions

Once the model is trained, you can use `app.py` to make predictions:

1. Modify the `example_features` in `app.py` with the features you want to test. Ensure the order of features is as follows:
    - `'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',`
    - `'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',`
    - `'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',`
    - `'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',`
    - `'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',`
    - `'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'`

2. Run `app.py` to get a prediction:
    ```bash
    python app.py
    ```
   The script will output whether the cancer is predicted to be `Benign` or `Malignant`.

## Example
- Make sure to update the example_features array in app.py with the new values you wish to test
- Here’s an example of using `app.py`:

```python
example_features = [
    14.5, 23.5, 90.3, 600.0, 0.1, 0.1, 0.2, 0.3, 0.2, 0.06,
    1.0, 1.2, 7.0, 80.0, 0.03, 0.04, 0.1, 0.2, 0.03, 0.04,
    14.0, 25.0, 85.0, 550.0, 0.09, 0.12, 0.18, 0.28, 0.19, 0.055
]

result = predict(example_features)
print(f'The prediction for the input features is: {result}')
