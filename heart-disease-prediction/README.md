# Heart Disease Prediction Using Logistic Regression

![img](https://github.com/IrfanKpm/mini-ml-projects/blob/main/images/heart.jpg)

This project utilizes a logistic regression model to predict heart disease based on various medical features. The dataset used for training the model is publicly available, and the model is saved and used for making predictions on new data.

## Project Overview

- **`modelbuild.py`**: This script trains a logistic regression model using the heart disease dataset and saves the trained model as a `.pkl` file.
- **`app.py`**: This script loads the saved model and allows users to input new data to predict whether a person has a healthy heart or defective heart.

## Files

- **`modelbuild.py`**: Script for training and saving the model.
- **`app.py`**: Script for loading the model and making predictions.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib scikit-learn
```


### Running the Model Training Script
```bash
python modelbuild.py
```

### Running the Prediction Script
```bash
python app.py
```
