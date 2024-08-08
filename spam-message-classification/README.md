# Spam Classifier
![img](https://github.com/IrfanKpm/mini-ml-projects/blob/main/images/spammail.png)
## Overview

This project involves building a spam classifier using Logistic Regression and TF-IDF Vectorizer. The model is trained to classify emails as either "spam" or "ham" (non-spam). The trained model and vectorizer are saved for later use with a FastAPI application.

## Approach

1. **Data Collection:** The dataset is sourced from a raw CSV file hosted on GitHub.
2. **Data Preparation:**
   - Missing values are filled with empty strings.
   - The 'Category' column is renamed to 'label' and mapped to binary values (0 for spam and 1 for ham).
3. **Feature Extraction:** TF-IDF Vectorizer is used to convert text data into numerical features.
4. **Model Training:** A Logistic Regression model is trained on the extracted features.
5. **Model Evaluation:** The model's accuracy is evaluated on both training and testing sets.
6. **Model Saving:** The trained model and vectorizer are saved using `joblib`.

## Model Used

- **Algorithm:** Logistic Regression
- **Feature Extraction:** TF-IDF Vectorizer
- **Training Accuracy:** 0.9677
- **Testing Accuracy:** 0.9668

## Files

- **`buildmodel.py`**: Script to build and save the model.
- **`spam_classifier_model.pkl`**: Trained Logistic Regression model.
- **`tfidf_vectorizer.pkl`**: TF-IDF Vectorizer used for feature extraction.

## Instructions

### Prerequisites

Ensure you have the required libraries installed. You can install them using:

```bash
pip install pandas numpy scikit-learn joblib
```
