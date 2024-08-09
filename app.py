import pickle
import numpy as np
import pandas as pd


# Load the saved model
with open('logistic_regression_model.pkl', 'rb') as file:
    saved_model = pickle.load(file)

# Feature names (matching the columns used during training)
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Corrected input data (13 features)
#input_data = (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0,2)
#input_data = input("data input > ")
print("\n[info] input data : age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal")
print("[eg]Enter data : 41 0 1 130 204 0 0 172 0 1.4 2 0 2\n")
input_data = tuple(input('Enter data : ').split())

  # Ensure this has 12 features
input_data_as_array = np.array(input_data).reshape(1, -1)


# Convert the input data to a DataFrame with feature names
input_data_df = pd.DataFrame(input_data_as_array, columns=feature_names)

# Predict using the saved model
prediction = saved_model.predict(input_data_df)

# Output the prediction result
if prediction[0] == 1:
    print("\n[~] The model predicts: Healthy heart.")
else:
    print("\n[!] The model predicts: Defective heart.")
