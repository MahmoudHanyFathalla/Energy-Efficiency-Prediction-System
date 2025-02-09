# Energy Efficiency Prediction System

## Overview
This project is an **Energy Efficiency Prediction System** that utilizes **machine learning** to predict the **heating load** required for buildings based on architectural and environmental features. The system is built using **Python** and leverages various data preprocessing techniques, machine learning algorithms, and feature engineering methods to improve prediction accuracy.

The project consists of:
- **Data Preprocessing**: Handling missing values, normalizing features, and cleaning data.
- **Machine Learning Model Training**: Using **Random Forest Regressor** to predict heating load.
- **Model Deployment**: Saving the trained model for future use in making predictions.
- **Feature Importance Analysis**: Identifying the most influential features affecting heating load.
- **Scalability & Performance Optimization**: Ensuring the model is efficient for large datasets.

## Features
- **Automated Data Cleaning**: Handles missing values and removes unnecessary columns.
- **Feature Normalization**: Scales data using **MinMaxScaler**.
- **Random Forest Regression Model**: Used for accurate predictions.
- **User Input Prediction**: Accepts new data points and predicts heating load.
- **Model Persistence**: Saves and reloads models for efficient deployment.
- **Feature Importance Ranking**: Determines which features have the most impact on heating load.

## Technologies Used
### Programming Language:
- **Python**

### Libraries:
- **Pandas** – For data manipulation and preprocessing.
- **NumPy** – For numerical computations.
- **Scikit-learn** – For machine learning model training and evaluation.
- **Joblib** – For saving and loading the trained model.
- **OpenPyXL** – For reading Excel files.
- **Matplotlib & Seaborn** – For visualizing feature importance and model performance.

## How It Works
### Training:
1. Loads the **Energy Efficiency Dataset**.
2. Cleans and normalizes the data.
3. Trains the **Random Forest Regressor**.
4. Evaluates the model's performance.
5. Saves the trained model and scaler for future use.
6. Analyzes feature importance.

#### Code Example:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_excel('Energy Efficiency Dataset.xlsx', sheet_name='sheet1', engine='openpyxl')
data = data.dropna()

# Feature selection
features = data[['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']]
target = data['Heating Load']

# Normalize features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

# Save model and scaler
joblib.dump(model, 'heating.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### Prediction:
1. Loads the saved model and scaler.
2. Accepts new input values.
3. Normalizes the input data.
4. Predicts the **Heating Load**.
5. Outputs the predicted result.

#### Code Example:
```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('heating.pkl')
scaler = joblib.load('scaler.pkl')

# New input data
new_data = np.array([0.64, 784, 343, 220.50, 3.5, 4, 0.25, 1]).reshape(1, -1)

# Normalize input
data_normalized = scaler.transform(new_data)

# Predict heating load
predicted_heating_load = model.predict(data_normalized)[0]
print(f"Predicted Heating Load: {predicted_heating_load}")
```

## Example Usage
1. **Training Phase**:
   ```sh
   python train.py
   ```
   - Cleans the dataset.
   - Trains the model.
   - Saves the model (`heating.pkl`) and scaler (`scaler.pkl`).

2. **Prediction Phase**:
   ```sh
   python use.py
   ```
   - Loads the trained model.
   - Accepts new input data.
   - Outputs the predicted heating load.

## Future Enhancements
- **Additional Feature Engineering**: Consider more variables that impact heating load.
- **Hyperparameter Tuning**: Optimize the Random Forest model further.
- **Web-based Interface**: Deploy as a web application for easy user interaction.
- **Multi-target Prediction**: Extend model to predict cooling load alongside heating load.
- **Integration with IoT Sensors**: Utilize real-time environmental data for improved accuracy.

## Conclusion
This **Energy Efficiency Prediction System** provides a reliable method for estimating heating requirements using machine learning. By leveraging **data preprocessing, feature selection, model persistence, and feature importance analysis**, the system offers accurate and scalable predictions for real-world applications.

---
Developed by **Mahmoud Hany**

