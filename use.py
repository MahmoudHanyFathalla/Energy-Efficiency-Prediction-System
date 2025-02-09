import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the saved model and scaler
model = joblib.load('heating.pkl')
scaler = joblib.load('scaler.pkl')  # Save scaler during training and load it here

# Step 2: Define new input data
# Example input: [Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution]
new_data = np.array([0.64, 784, 343, 220.50, 3.5, 4, 0.25, 1]).reshape(1, -1)

# Step 3: Normalize the input data
normalized_data = scaler.transform(new_data)

# Step 4: Predict the Heating Load
predicted_heating_load = model.predict(normalized_data)[0]

# Step 5: Output the result
print(f"Predicted Heating Load: {predicted_heating_load}")
